# 去重
from tqdm import tqdm
import cv2
import numpy as np
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import warnings
import os

warnings.filterwarnings("ignore")

path = 'D:/im'  # 图片路径
mid_ssim = 90  # 中间最大运动幅度
min_vec = 0.8  # 双边最小运动幅度
max_ssim = 99.9  # 重复帧阈值

print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]


def ssim(i0, i1):
    return compare_ssim(i0, i1, multichannel=True) * 100


pbar = tqdm(total=len(frames))
duplicate = []
lf = frames[0]
for i in range(1, len(frames)):
    f = frames[i]
    # 两两对比，ssim值大于max_ssim的，辨别为重复帧
    if ssim(lf, f) > max_ssim:
        duplicate.append(i)
    lf = f
    pbar.update(1)
for x in duplicate:
    try:
        del frames[x]
        os.remove(LabData[x])
        del LabData[x]
    except:
        print('pass at {}'.format(x))

# 去除一拍二
print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]

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
    if x2 - x1 > min_vec and x2 - x3 > min_vec and x2 > mid_ssim:
        duplicate.append([i - 1, i, i + 1, i + 2])
    I0 = I1
    pbar.update(1)
    i += 1
for x in duplicate:
    try:
        del frames[x[1]]
        os.remove(LabData[x[1]])  # i0,i1,i2,i3 这里移除的是i1帧 （也可以选择i2帧）
        del LabData[x[1]]
    except:
        print('pass at {}'.format(x[1]))

# 去除一拍三
print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]


def mask_denoise(mask):
    labels = measure.label(mask, connectivity=2)  # 8连通区域标记
    properties = measure.regionprops(labels)
    return np.in1d(labels, [0]).reshape(labels.shape)


# 使用五帧
def compare(i0, i1, i2, i3, i4):
    i1_ = i1.mean(2).astype("uint8")
    i2_ = i2.mean(2).astype("uint8")
    i3_ = i3.mean(2).astype("uint8")
    i1_ = cv2.medianBlur(i1_, 3)
    i2_ = cv2.medianBlur(i2_, 3)
    i3_ = cv2.medianBlur(i3_, 3)
    # 中间三帧来做mask
    mask = np.stack([i1_, i2_, i3_]).var(0)
    max_value = np.max(mask)
    ret, mask = cv2.threshold(mask, 0.005 * max_value, max_value, cv2.THRESH_BINARY)
    mask = mask_denoise(mask)
    i0[mask < 0.5] = 255
    i1[mask < 0.5] = 255
    i2[mask < 0.5] = 255
    i3[mask < 0.5] = 255
    i4[mask < 0.5] = 255
    # 返回处理后的5帧
    return [i0, i1, i2, i3, i4]


duplicate = []
i0 = frames[0]
pbar = tqdm(total=len(frames))
i = 1
while i < len(LabData) - 3:
    i1 = frames[i]
    i2 = frames[i + 1]
    i3 = frames[i + 2]
    i4 = frames[i + 3]
    c = compare(i0, i1, i2, i3, i4)
    # (ssim(i1,i2) + ssim(i2,i3)) / 2
    m = (ssim(c[1], c[2]) + ssim(c[2], c[3])) / 2
    # ssim(i0,i1)
    l = ssim(c[0], c[1])
    # ssim(i3,i4)
    r = ssim(c[3], c[4])
    # 估计三帧ssim值(i1,i2,i3) - ssim(i0,i1) > min_vec AND 估计三帧ssim值 - ssim(i3,i4) > min_vec
    if m - l > min_vec and m - r > min_vec and m > mid_ssim:
        duplicate.append([i - 1, i, i + 1, i + 2, i + 3])
    i0 = i1
    pbar.update(1)
    i += 1
for x in duplicate:
    try:
        os.remove(LabData[x[1]])  # 这里选择移除旁边两帧，具体可以自己选择
        os.remove(LabData[x[3]])
    except:
        print('pass at {} {}'.format(x[1], x[3]))
