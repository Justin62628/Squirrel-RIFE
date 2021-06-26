# 去除重复帧

from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os

warnings.filterwarnings("ignore")

path = 'D:/im'  # 图片路径

print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f, 0), (256, 256)) for f in LabData]


def diff(i0, i1):
    return cv2.absdiff(i0, i1).mean()


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
