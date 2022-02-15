import time

from torch.nn.parallel import DistributedDataParallel as DDP

from RIFE.IFNet_HDv4 import *
from RIFE.loss import *
from Utils.StaticParameters import RGB_TYPE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=0, ada=False, local_rank=-1):
        self.tta = tta
        self.ada = ada
        self.forward_ensemble = forward_ensemble
        self.use_multi_cards = use_multi_cards
        self.device_count = torch.cuda.device_count()
        self.flownet = IFNet()
        self.device()
        self.version = 4.0
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location='cpu')),
                                             False)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))

    def get_auto_scale(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        scale = self.flownet.get_auto_scale(imgs)
        return scale

    def calculate_prediction(self, img0, img1, scale, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, timestep, scale_list, ada=self.ada)
        return merged[3]

    def inference(self, img0, img1, scale=1.0, n=1):
        multiple = n + 1
        inferenced = list()
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for frame_index, t_value in enumerate(t):
            prediction = self.calculate_prediction(img0, img1, scale, t_value)
            prediction = (prediction[0] * RGB_TYPE.SIZE).float().cpu().numpy()
            inferenced.append(prediction)
            del prediction
        return inferenced


if __name__ == '__main__':
    _img0 = torch.zeros(1, 3, 256, 256).float().to(device)
    _img1 = torch.tensor(np.random.normal(
        0, 1, (1, 3, 256, 256))).float().to(device)
    _imgs = torch.cat((_img0, _img1), 1)
    model = Model(True, True, 3)
    model.load_model(r"D:\60-fps-Project\Projects\RIFE GUI\train_log\official_4.0", -1)
    model.eval()
    _t = time.time()
    _inferenced = model.inference(_img0, _img1, n=2)
    print(_inferenced[0].shape)
    print(round(time.time() - _t, 2))
