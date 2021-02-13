import os
import cv2
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue, Empty
import subprocess as sp
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0')
parser.add_argument('--scene', dest='scene', type=float, default=30)
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--vector', dest='vector', type=int, default=1)

parser.add_argument('--direct', dest='direct', action='store_true')
parser.add_argument('--ffmpeg', dest='ffmpeg', type=str, default="ffmpeg.exe")
parser.add_argument('--read_fps', dest='read_fps', type=float, default=60.00)
parser.add_argument('--out_fps', dest='out_fps', type=float, default=60.00)
parser.add_argument('--crf', dest='crf', type=int, default=16)
parser.add_argument('--audio', dest='audio', type=str, default="")

###
parser.add_argument('--batch_size', dest='batch_size', type=int, default=50)
###

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

print("loading train_log...")
if args.gpu_id != '-1':
    # torch.cuda.set_device()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
gpuNum = torch.cuda.device_count()

if args.gpu_id == '-1':
    from model_cpu.RIFE_HDv2 import Model

    model = Model()
    model.load_model('{}\\train_log'.format(dname), -1)
    model.eval()
    model.device()
else:
    from model.RIFE_HDv2 import Model

    for gpuCount in range(gpuNum):
        model_list = []
        torch.cuda.set_device(gpuCount)
        model_list.append(Model())
        model_list[gpuCount].load_model('{}\\train_log'.format(dname), -1)
        model_list[gpuCount].eval()
        model_list[gpuCount].device()

print("train_log loaded")

videogen = []
for f in os.listdir(args.img):
    if 'png' in f:
        videogen.append(f)
tot_frame = len(videogen)
videogen.sort(key=lambda x: int(x[:-4]))
lastframe = cv2.imread(os.path.join(args.img, videogen[0]))[:, :, ::-1].copy()
# videogen = videogen[1:]
h, w, _ = lastframe.shape


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(Dataset):
    def __init__(self, img, videogen, tot_frame):
        self.path = img
        self.videogen = videogen
        self.tot_frame = tot_frame

    def __getitem__(self, index):
        frame = self.videogen[index]
        pic = cv2.imread(os.path.join(self.path, frame))
        if pic is not None:
            data = pic[:, :, ::-1].copy()
        elif index != 0:
            data = cv2.imread(os.path.join(self.path, self.videogen[index - 1]))[:, :, ::-1].copy()
        else:
            data = None
        return data

    def __len__(self):
        return tot_frame


dataloader = GetLoader(args.img, videogen, tot_frame)
datas = DataLoaderX(dataloader, batch_size=args.batch_size, shuffle=False, drop_last=False,
                   num_workers=0)

pipe = 0
if args.direct:
    command = [args.ffmpeg,
               '-y',
               '-f', 'rawvideo',
               '-c:v', 'rawvideo',
               '-s', '{}x{}'.format(w, h),
               '-pix_fmt', 'rgb24',
               '-r', '{}'.format(args.read_fps),
               '-i', '-',
               '-c:v', 'h264',
               '-pix_fmt', 'yuv420p',
               '-crf', '{}'.format(args.crf),
               '-r', '{}'.format(args.out_fps),
               '{}'.format(args.output)]
    if args.audio != "":
        command = [args.ffmpeg,
                   '-y',
                   '-f', 'rawvideo',
                   '-c:v', 'rawvideo',
                   '-s', '{}x{}'.format(w, h),
                   '-pix_fmt', 'rgb24',
                   '-r', '{}'.format(args.read_fps),
                   '-i', '-',
                   '-i', '{}'.format(args.audio),
                   '-c:v', 'h264',
                   '-pix_fmt', 'yuv420p',
                   '-crf', '{}'.format(args.crf),
                   '-r', '{}'.format(args.out_fps),
                   '-c:a', 'aac',
                   '-b:a', '320k',
                   '{}'.format(args.output)]
    pipe = sp.Popen(command, stdin=sp.PIPE, bufsize=-1)


def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if args.direct:
            pipe.stdin.write(item.tobytes())
            pipe.stdin.flush()
        else:
            cv2.imwrite('{}/{:0>9d}.png'.format(args.output, cnt), item[:, :, ::-1])
        cnt += 1


# def build_read_buffer(user_args, read_buffer, videogen):
#     for frame in videogen:
#         if not user_args.img is None:
#             frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
#         read_buffer.put(frame)
#     read_buffer.put(None)

def make_inference(I0, I1, exp, model):
    middle = model.inference(I0, I1, args.UHD)
    if exp == 1:
        return [middle]
    first_half = make_inference(I0, middle, exp=exp - 1, model=model)
    second_half = make_inference(middle, I1, exp=exp - 1, model=model)
    return [*first_half, middle, *second_half]


def make_inference_back(I0, I1, exp, model):
    middle = model.inference(I1, I0, args.UHD)
    if exp == 1:
        return [middle]
    first_half = make_inference_back(I0, middle, exp=exp - 1, model=model)
    second_half = make_inference_back(middle, I1, exp=exp - 1, model=model)
    return [*first_half, middle, *second_half]


if args.UHD:
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
else:
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
write_buffer = Queue(1500)
# read_buffer = Queue(maxsize=10000)
# _thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

cal_count = 0
# while True:
for batch_step, data in enumerate(datas):
    for j in range(len(data)):
        if args.gpu_id != '-1':
            device = torch.device("cuda", index=(cal_count % gpuNum))
            model = model_list[cal_count % gpuNum]
        if (batch_step == 0) & (j == 0):
            I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(
                0).float() / 255.
            I1 = F.pad(I1, padding)
            pbar.update(1)
            continue
        frame = data[j].numpy()
        # if frame is None:
        #     break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(
            0).float() / 255.
        I1 = F.pad(I1, padding)
        # diff = abs(I0-I1).mean()*100
        diff = cv2.absdiff(lastframe[:, :, ::-1], frame[:, :, ::-1]).mean()
        if diff > args.scene:
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
                alpha += step
                beta = 1 - alpha
                output.append(torch.from_numpy(np.transpose(
                    (cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()),
                    (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            # output = []
            # for i in range((2 ** args.exp) - 1):
            #    output.append(I0)
        else:
            cal_count = cal_count + 1
            if args.vector == 1:
                output = make_inference(I0, I1, args.exp, model)
            else:
                output = make_inference_back(I0, I1, args.exp, model)
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame
write_buffer.put(lastframe)
print("等待帧写入完成...")
import time

while (not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
