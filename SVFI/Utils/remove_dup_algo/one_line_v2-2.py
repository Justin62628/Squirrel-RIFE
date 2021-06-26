from tqdm import tqdm
import cv2
import warnings
import os

warnings.filterwarnings("ignore")

path = 'D:/experiment/AVFDU/origin'  # 图片路径
death_diff = 20  # 中间最大运动幅度

print('loading data to ram...')  # 将数据载入到内存中，加速运算
LabData = [os.path.join(path, f) for f in os.listdir(path)]  # 记录文件名用
frames = [cv2.resize(cv2.imread(f), (256, 256)) for f in LabData]  # 帧


def diff(i0, i1):
    # CannyEdge和absdiff作为判断方法
    return cv2.Canny(cv2.absdiff(i0, i1), 100, 200).mean()


max_epoch = 7  # 一直去除到一拍N，N为max_epoch
queue_size = 3  # 吞入帧数
opt = []  # 已经被标记，识别的帧
I0 = frames[0]  # 第一帧
pbar = tqdm(total=max_epoch)  # 总轮数
for _ in range(max_epoch):
    queue_size += 1  # 加长队列长度
    Icount = queue_size - 1  # 输入帧数
    Current = []  # 该轮被标记的帧
    for i in range(1, len(LabData) - Icount):  # - Icount
        c = [frames[p + i] for p in range(queue_size)]  # 读取queue_size帧图像
        l = diff(c[0], c[1])  # 左侧diff
        r = diff(c[len(c) - 2], c[-1])  # 右侧diff
        m = 0  # diff中值
        m += sum(diff(c[x], c[x + 1]) for x in range(1, len(c) - 2))  # 叠加中值
        m /= len(c) - 3  # 取平均
        if l > m and r > m and m < death_diff:  # 满足约束条件
            Current.append(i)  # 加入标记序号
    opted = len(opt)  # 记录opt长度
    for x in Current:
        if x - 1 not in opt and x + 1 not in opt and x not in opt:  # 优化:该轮一拍N不可能出现在上一轮中
            for t in range(queue_size - 3):
                opt.append(t + x)
    pbar.update(1)  # 完成一轮
    if len(opt) == opted:  # 如果相等则证明已经标记完了所有帧，不存在更多的节拍数
        break
delgen = sorted(set(opt))  # 需要删除的帧
for d in delgen:
    try:
        os.remove(LabData[d + 1])
    except:
        print('pass')
pbar.close()
