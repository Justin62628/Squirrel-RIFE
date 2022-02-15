import time

from torch.nn.parallel import DistributedDataParallel as DDP

from RIFE.IFNet_v7_multi import *
from RIFE.loss import *
from RIFE.refine_v6 import *

device = torch.device("cuda")


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=0, local_rank=-1):
        self.tta = tta
        self.forward_ensemble = forward_ensemble
        self.use_multi_cards = use_multi_cards
        self.device_count = torch.cuda.device_count()
        if self.device_count > 1 and self.use_multi_cards:
            self.flownet = nn.DataParallel(IFNet_m())
        else:
            self.flownet = IFNet_m()
        self.device()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))

    def calculate_prediction(self, img0, img1, scale, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = \
            self.flownet(imgs, scale_list, timestep=timestep, ensemble=self.forward_ensemble)
        return merged[2]

    def inference(self, img0, img1, scale=1.0, n=1):
        multiple = n + 1
        inferenced = list()
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for frame_index, t_value in enumerate(t):
            inferenced.append(self.calculate_prediction(img0, img1, scale, t_value))
        return inferenced


if __name__ == '__main__':
    _img0 = torch.zeros(1, 3, 256, 256).float().to(device)
    _img1 = torch.tensor(np.random.normal(
        0, 1, (1, 3, 256, 256))).float().to(device)
    _imgs = torch.cat((_img0, _img1), 1)
    model = Model(True, True, 3)
    model.load_model(r"D:\60-fps-Project\Projects\RIFE GUI\train_log\official_v7_multi", -1)
    model.eval()
    _t = time.time()
    _inferenced = model.inference(_img0, _img1, n=2)
    print(_inferenced[0].shape)
    print(round(time.time() - _t, 2))
