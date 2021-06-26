import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import time
from queue import Queue

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='对图片序列进行补帧')
parser.add_argument('--img', dest='img', type=str, default='input', help='图片目录')
parser.add_argument('--output', dest='output', type=str, default='out', help='保存目录')
parser.add_argument('--start', dest='start', type=int, default=0, help='从第start张图片开始补帧')
parser.add_argument('--device_id', dest='device_id', type=int, default=0, help='设备ID')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='模型目录')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='FP16速度更快，质量略差')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='4K时建议0.5')
parser.add_argument('--rbuffer', dest='rbuffer', type=int, default=0, help='读写缓存')
parser.add_argument('--wthreads', dest='wthreads', type=int, default=4, help='写入线程')

parser.add_argument('--scene', dest='scene', type=float, default=50, help='场景识别阈值')
parser.add_argument('--rescene', dest='rescene', type=str, default="mix", help="copy/mix   帧复制/帧混合")
parser.add_argument('--times', dest='times', type=int, default=2, help='补x倍帧率')


def build_interpolation_guide(path, times=2):
    guide = []
    LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
    frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]  # 帧
    times -= 1
    inter_frames = times * 2
    for i in range(0, len(frames) - 2, 2):
        f0 = frames[i]
        f1 = frames[i + 1]
        f2 = frames[i + 2]
        x1 = cv2.absdiff(f0, f1).mean()
        x2 = cv2.absdiff(f1, f2).mean()
        dt = x1 + x2
        if dt > 0:
            i1 = int(x1 / dt * inter_frames)
            i2 = int(x2 / dt * inter_frames)
            z = inter_frames - i1 - i2
            if z != 0:
                if i1 > i2:
                    i1 += z
                else:
                    i2 += z
        else:
            i1 = int(inter_frames / 2)
            i2 = inter_frames - i1
        guide.append(i1)
        guide.append(i2)
        f0 = f2
    return guide


args = parser.parse_args()
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
spent = time.time()

if not os.path.exists(args.output) and not args.out_video:
    os.mkdir(args.output)

if args.device_id != -1:
    device = torch.device("cuda")
    torch.cuda.set_device(args.device_id)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    try:
        from model.RIFE_HDv2 import Model

        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        try:
            from model.RIFE_HDv3 import Model

            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model")
        except:
            from model.RIFE_HD import Model

            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v1.x HD model")
else:
    device = torch.device("cpu")
    try:
        from model_cpu.RIFE_HDv2 import Model

        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v2.x HD model.")
    except:
        try:
            from model_cpu.RIFE_HDv3 import Model

            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model")
        except:
            from model_cpu.RIFE_HD import Model

            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v1.x HD model")

model.eval()
model.device()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

start = args.start

videogen = [f for f in os.listdir(args.img)]
tot_frame = len(videogen)
if start != 0:
    templist = []
    pos = start - 1
    end = len(videogen)
    while pos != end:
        templist.append(videogen[pos])
        pos = pos + 1
    videogen = templist
passed = tot_frame - len(videogen)
videogen.sort()
lastframe = cv2.imdecode(np.fromfile(os.path.join(args.img, videogen[0]), dtype=np.uint8), 1)[:, :, ::-1].copy()
videogen = videogen[1:]
h, w, _ = lastframe.shape


def clear_write_buffer(user_args, write_buffer):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        num = item[0]
        content = item[1]
        cv2.imencode('.png', content[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(user_args.output, num))


def build_read_buffer(dir_path, read_buffer, videogen):
    try:
        for frame in videogen:
            frame = cv2.imdecode(np.fromfile(os.path.join(dir_path, frame), dtype=np.uint8), 1)[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def pad_image(img, padding):
    if (args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


def make_inference(im0, im1, n):
    I0 = torch.from_numpy(np.transpose(im0, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = torch.from_numpy(np.transpose(im1, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0, padding)
    I1 = pad_image(I1, padding)
    global model
    middle = model.inference(I0, I1, args.scale)
    I0 = 0
    I1 = I0
    mid = (((middle[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w][:, :, ::-1][:, :, ::-1].copy()
    if n == 1:
        return [mid]
    first_half = make_inference(im0, mid, n=n // 2)
    second_half = make_inference(mid, im1, n=n // 2)
    if n % 2:
        return [*first_half, mid, *second_half]
    else:
        return [*first_half, *second_half]


pbar = tqdm(total=tot_frame)
pbar.update(passed)
read_buffer = Queue(maxsize=args.rbuffer)
write_buffer = Queue(maxsize=args.rbuffer)
frame_writer = 0
_thread.start_new_thread(build_read_buffer, (args.img, read_buffer, videogen))
for _ in range(args.wthreads):
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer))

cnt = 0
cnt = 0 if start == 0 else (start - 1) * (2 ** args.exp) + 1
cnt += 1

h, w, _ = lastframe.shape
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

guide = build_interpolation_guide(args.img, args.times)
pos = 0
while True:
    frame = read_buffer.get()
    if frame is None:
        break
    output = make_inference(lastframe, frame, guide[pos]) if guide[pos] > 0 else []
    write_buffer.put([cnt, lastframe])
    cnt += 1
    lo = len(output)
    for x in range(lo):
        write_buffer.put([cnt, output[x]])
        cnt += 1
    pbar.update(1)
    lastframe = frame
    pos += 1
write_buffer.put([cnt, lastframe])
if not args.out_video:
    while (not os.path.exists('{}/{:0>9d}.png'.format(args.output, cnt))):
        time.sleep(1)
pbar.close()
print("spent {}s".format(time.time() - spent))
