import threading
from tqdm import tqdm
import cv2
import warnings
import os
import numpy as np

warnings.filterwarnings("ignore")


def sobel(src):
    src = cv2.GaussianBlur(src, (3, 3), 0)
    src = cv2.fastNlMeansDenoising(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, -1, 3, 0, ksize=5)
    grad_y = cv2.Sobel(gray, -1, 0, 3, ksize=5)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)


def diff(i0, i1):
    return cv2.absdiff(i0, i1).mean()


def diff_canny(i0, i1):
    return cv2.Canny(cv2.absdiff(i0, i1), 100, 200).mean()


def predict_scale(i0, i1):
    w, h, _ = i0.shape
    diff = cv2.Canny(sobel(cv2.absdiff(i0, i1)), 100, 200)
    mask = np.where(diff != 0)
    try:
        xmin = min(list(mask)[0])
    except:
        xmin = 0
    try:
        xmax = max(list(mask)[0]) + 1
    except:
        xmax = w
    try:
        ymin = min(list(mask)[1])
    except:
        ymin = 0
    try:
        ymax = max(list(mask)[1]) + 1
    except:
        ymax = h
    W = xmax - xmin
    H = ymax - ymin
    S0 = w * h
    S1 = W * H
    return -2 * (S1 / S0) + 3


# 亮度均衡化
def yuv(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def calc_flow_distance(i0, i1, use_flow=True):
    if not use_flow:
        return diff_canny(i0, i1)
    prev_gray = cv2.cvtColor(i0, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    flow0 = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray,
                                         flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                         winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
    flow1 = cv2.calcOpticalFlowFarneback(curr_gray, prev_gray,
                                         flow=None, pyr_scale=0.5, levels=1, iterations=20,
                                         winsize=15, poly_n=5, poly_sigma=1.1, flags=0)
    flow = (flow0 - flow1) / 2
    x = flow[:, :, 0]
    y = flow[:, :, 1]
    return np.linalg.norm(x) + np.linalg.norm(y)


path = 'D:/experiment/AVFDU/origin'  # 图片路径

LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [yuv(cv2.resize(cv2.imread(f), (256, 256))) for f in LabData]

print('destatic abs frames...')
pbar = tqdm(total=len(frames))
delgen = []
lf = frames[0]
for i in range(1, len(frames)):
    f = frames[i]
    if diff(lf, f) == 0:
        delgen.append(LabData[i])
    lf = f
    pbar.update(1)
for x in delgen:
    os.remove(x)

LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [yuv(cv2.resize(cv2.imread(f), (256, 256))) for f in LabData]

static = []
print('build canny static frame list...')
pbar = tqdm(total=len(frames))
lf = frames[0]
for i in range(1, len(frames)):
    f = frames[i]
    if diff_canny(lf, f) == 0:
        static.append(i - 1)
    lf = f
    pbar.update(1)

LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [yuv(cv2.resize(cv2.imread(f), (32, 32))) for f in LabData]

print('build one beta x frame list...')
max_epoch = 3  # 一直去除到一拍N，N为max_epoch（不建议超过3）
opt = []  # 已经被标记，识别的帧
I0 = frames[0]  # 第一帧
pbar = tqdm(total=max_epoch)  # 总轮数
use_flow = True  # 使用光流
# value_scale = 1.0 # 乘光流距离(默认2.0即可)

for queue_size, _ in enumerate(range(1, max_epoch), start=4):
    Icount = queue_size - 1  # 输入帧数
    Current = []  # 该轮被标记的帧
    i = 1
    while (i < len(LabData) - Icount):
        # for i in range(1,len(LabData) - Icount):
        c = [frames[p + i] for p in range(queue_size)]  # 读取queue_size帧图像
        first_frame = c[0]
        last_frame = c[-1]
        count = 0
        for step in range(1, queue_size - 2):
            pos = 1
            while (pos + step <= queue_size - 2):
                m0 = c[pos]
                m1 = c[pos + step]
                d0 = calc_flow_distance(first_frame, m0, use_flow)
                d1 = calc_flow_distance(m0, m1, use_flow)
                d2 = calc_flow_distance(m1, last_frame, use_flow)
                value_scale = predict_scale(m0, m1)
                if value_scale * d1 < d0 and value_scale * d1 < d2:
                    count += 1
                pos += 1
        if count == (queue_size * (queue_size - 5) + 6) / 2:
            Current.append(i)  # 加入标记序号
            i += queue_size - 3
            # print(i-2,d0,d1,d2)
        i += 1
    opted = len(opt)  # 记录opt长度
    for x in Current:
        # if x - 1 not in opt and x + 1 not in opt and x not in opt: # 弃用
        if x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
            for t in range(queue_size - 3):
                opt.append(t + x + 1)
    pbar.update(1)  # 完成一轮

print('concat result...')
opt.extend(static)
delgen = sorted(set(opt))  # 需要删除的帧
for d in delgen:
    try:
        os.remove(LabData[d])
    except:
        print('pass')
pbar.close()
