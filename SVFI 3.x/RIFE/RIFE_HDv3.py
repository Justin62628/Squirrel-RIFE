import time

from torch.nn.parallel import DistributedDataParallel as DDP

from RIFE.IFNet_HDv3 import *
from RIFE.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=0, ada=False, local_rank=-1):
        self.tta = tta
        self.forward_ensemble = forward_ensemble
        self.use_multi_cards = use_multi_cards
        self.device_count = torch.cuda.device_count()
        self.ada = ada
        if self.device_count > 1 and self.use_multi_cards:
            self.flownet = nn.DataParallel(IFNet())
        else:
            self.flownet = IFNet()
        self.device()
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

    def calculate_prediction(self, img0, img1, scale):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        merged = self.flownet(imgs, scale_list, ensemble=self.forward_ensemble, ada=self.ada)[0][2]
        return merged

    def TTA_FRAME(self, img0, img1, iter_time=2, scale=1.0):
        if iter_time != 0:
            img0 = self.calculate_prediction(img0, img1, scale)
            return self.TTA_FRAME(img0, img1, iter_time=iter_time - 1, scale=scale)
        else:
            return img0

    def inference(self, img0, img1, scale=1.0, iter_time=2):
        if self.tta == 0:
            return self.TTA_FRAME(img0, img1, 1, scale)
        elif self.tta == 1:  # side_vector
            LX = self.TTA_FRAME(img0, img1, iter_time, scale)
            RX = self.TTA_FRAME(img1, img0, iter_time, scale)
            return self.TTA_FRAME(LX, RX, 1, scale)
        elif self.tta == 2:  # mid_vector
            mid = self.TTA_FRAME(img0, img1, 1, scale)
            LX = self.TTA_FRAME(img0, mid, iter_time, scale)
            RX = self.TTA_FRAME(mid, img1, iter_time, scale)
            return self.TTA_FRAME(LX, RX, 1, scale)
        elif self.tta == 3:  # mix_vector
            mid = self.TTA_FRAME(img0, img1, 1, scale)
            LX = self.TTA_FRAME(img0, mid, iter_time, scale)
            RX = self.TTA_FRAME(mid, img1, iter_time, scale)
            m1 = self.TTA_FRAME(LX, RX, 1, scale)
            LX = self.TTA_FRAME(img0, img1, iter_time, scale)
            RX = self.TTA_FRAME(img1, img0, iter_time, scale)
            m2 = self.TTA_FRAME(LX, RX, 1, scale)
            return self.TTA_FRAME(m1, m2, 1, scale)


if __name__ == '__main__':
    _img0 = torch.zeros(1, 3, 256, 256).float().to(device)
    _img1 = torch.tensor(np.random.normal(
        0, 1, (1, 3, 256, 256))).float().to(device)
    _imgs = torch.cat((_img0, _img1), 1)
    model = Model(True, True, 3)
    model.load_model(r"D:\60-fps-Project\Projects\RIFE GUI\train_log\official_v6", -1)
    model.eval()
    _t = time.time()
    print(model.inference(_img0, _img1).shape)
    print(round(time.time() - _t, 2))
