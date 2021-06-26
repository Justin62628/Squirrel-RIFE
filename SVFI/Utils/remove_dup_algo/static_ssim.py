# 去除重复帧

from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os

warnings.filterwarnings("ignore")

path = 'D:/im'  # 图片路径
min_ssim = 99.9

print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]


def ssim(i0, i1):
    return compare_ssim(i0, i1, multichannel=True) * 100


pbar = tqdm(total=len(frames))
delgen = []
lf = frames[0]
for i in range(1, len(frames)):
    f = frames[i]
    # 两两对比，ssim值大于max_ssim的，辨别为重复帧
    if ssim(lf, f) > min_ssim:
        delgen.append(LabData[i])
    lf = f
    pbar.update(1)

for x in delgen:
    os.remove(x)
