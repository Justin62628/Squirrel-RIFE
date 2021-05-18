import numpy  as np
import math


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))
    return mae


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom


## use the scikit package
from PIL import Image

# img1 = np.array(Image.open(r"G:\RIFE-Interpolation\Frozen.II\001\interp\00001381.png"))
# img2 = np.array(Image.open(r"G:\RIFE-Interpolation\Frozen.II\001\frames\00000345.png"))
"""ori_2 - 7 96.5, 
ssim: 0.9650446164861856
psnr: 38.92041798971462
mae: 37.76374785381408
mse: 8.337552121658081

ori_1 - 7
ssim: 0.9657414935836779
psnr: 38.87744264391472
mae: 39.77276490066225
mse: 8.420465415746873
"""
from skimage.metrics import structural_similarity as ssim

# ssim(img1, img2)  # for gray image
# print(f"ssim: {ssim(img1, img2, multichannel=True)}")  ## for rgb/
# print(f"psnr: {psnr(img1, img2)}")  ## for rgb/
# print(f"mae: {mae(img1, img2)}")  ## for rgb/
# print(f"mse: {mse(img1, img2)}")  ## for rgb/

"""
其中，MSE表示当前图像X和参考图像Y的均方误差（Mean Square Error），H、W分别为图像的高度和宽度；n为每像素的比特数，一般取8，即像素灰阶数为256. 
PSNR的单位是dB，数值越大表示失真越小/。
PSNR是最普遍和使用最为广泛的一种图像客观评价指标，然而它是基于对应像素点间的误差，即基于误差敏感的图像质量评价。
"""
import os

def compare_folder(input1, input2):
    fd = os.listdir(input1)
    fd.reverse()
    for f1 in fd:
        f2 = os.path.join(input2, f1)
        img1_ = np.array(Image.open(f1))
        img2_ = np.array(Image.open(f2))
        print(f"SSIM {f1}: {ssim(img1_, img2_, multichannel=True)}\n")

i1 = r"G:\RIFE-Interpolation\STNC-debug\frames"
i2 = r"G:\RIFE-Interpolation\STNC-debug\frames[con]"
os.chdir(i1)
compare_folder(i1, i2)
