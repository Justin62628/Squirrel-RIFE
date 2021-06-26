# 去除一拍二

from tqdm import tqdm
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os
from skimage import measure

warnings.filterwarnings("ignore")

max_vec = 90
path = 'D:/im'  # 图片路径

print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]


def ssim(i0, i1):
    return compare_ssim(i0, i1, multichannel=True) * 100


duplicate = []  # 用于存放表示一拍二的多组四帧列表
I0 = frames[0]
pbar = tqdm(total=len(frames))
i = 1
while i < len(LabData) - 2:
    # i0,i1,i2,i3为输入帧
    I1 = frames[i]
    I2 = frames[i + 1]
    I3 = frames[i + 2]
    #   i0,i1  i1,i2   i2,i3   分别对比的到ssim值，i1,i2最为一个整体
    x1 = ssim(I0, I1)
    x2 = ssim(I1, I2)
    x3 = ssim(I2, I3)
    #   左侧ssim - 中间值(i1,i2) > 最小运动幅度     中间值 - 右侧ssim > 最小运动幅度
    if x2 > x1 and x2 > x3 and x2 > max_vec:
        duplicate.append([LabData[i - 1], LabData[i], LabData[i + 1], LabData[i + 2]])  # i-1,i,i+1,i+2分别为i0,i1,i2,i3
    I0 = I1
    pbar.update(1)
    i += 1
for x in duplicate:
    try:
        os.remove(x[1])  # i0,i1,i2,i3 这里移除的是i1帧 （也可以选择i2帧）
    except:
        print('pass:{}'.format(x[2]))
