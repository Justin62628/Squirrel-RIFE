from torch.nn.parallel import DistributedDataParallel as DDP

from RIFE.IFNet_HDv2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


c = 32


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv0 = Conv2(3, c)
        self.conv1 = Conv2(c, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv0(x)
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear",
                             align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.conv0 = Conv2(10, c)
        self.down0 = Conv2(c, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
        s0 = self.down0(x)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=0, ada=False, local_rank=-1):
        self.forward_ensemble = forward_ensemble
        self.tta = tta
        self.use_multi_cards = use_multi_cards
        self.ada = ada
        if self.use_multi_cards:
            self.flownet = nn.DataParallel(IFNet())
            self.contextnet = nn.DataParallel(ContextNet())
            self.fusionnet = nn.DataParallel(FusionNet())
        else:
            self.flownet = IFNet()
            self.contextnet = ContextNet()
            self.fusionnet = FusionNet()
        self.device()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[
                local_rank], output_device=local_rank)
            self.contextnet = DDP(self.contextnet, device_ids=[
                local_rank], output_device=local_rank)
            self.fusionnet = DDP(self.fusionnet, device_ids=[
                local_rank], output_device=local_rank)

    def eval(self):
        self.flownet.eval()
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank):
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
            self.flownet.load_state_dict(
                convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
            self.contextnet.load_state_dict(
                convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))

    def save_model(self, path, rank):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))
            torch.save(self.contextnet.state_dict(), '{}/contextnet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/unet.pkl'.format(path))

    def get_auto_scale(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        scale = self.flownet.get_auto_scale(imgs)
        return scale

    def predict(self, imgs, flow, training=True, flow_gt=None):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
        else:
            return pred

    def calculate_flow(self, img0, img1, scale):
        imgs = torch.cat((img0, img1), 1)
        flow, _ = self.flownet(imgs, scale, ensemble=self.forward_ensemble, ada=self.ada)
        if self.forward_ensemble:
            pass
        return flow, imgs

    def calculate_prediction(self, img0, img1, scale):
        flow, imgs = self.calculate_flow(img0, img1, scale)
        prediction = self.predict(imgs, flow, training=False)
        return prediction

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
    model.eval()
    print(model.inference(_img0, _img1).shape)
