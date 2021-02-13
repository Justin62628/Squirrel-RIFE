import os
import cv2
import torch
import argparse
import shutil
import numpy as np
import json
import codecs
from torch.nn import functional as F
from tqdm import tqdm
import warnings
import _thread
from queue import Queue, Empty
import threading
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

warnings.filterwarnings("ignore")
    
parser = argparse.ArgumentParser(description='Interpolation by json file log')
parser.add_argument('--json', dest='json', type=str, default=None)
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--vector', dest='vector', type=int, default=1)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=500)
args = parser.parse_args()

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

# def build_read_buffer(imgs, read_buffer, videogen):
#     for frame in videogen:
#         if not imgs is None:
#             frame = cv2.imread(os.path.join(imgs, frame))[:, :, ::-1].copy()
#         read_buffer.put(frame)
#     read_buffer.put(None)

def imwrite(path,content):
    cv2.imwrite("{}.{}".format(path,"png"), content)

def interpolate(imgs,out):
    thread_list = []
    cnt = 0
    videogen = []
    for f in os.listdir(imgs):
        if 'png' in f:
            videogen.append(f)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(imgs, videogen[0]))[:, :, ::-1].copy()
    # videogen = videogen[1:]
    h, w, _ = lastframe.shape

    # 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
    class GetLoader(Dataset):
        def __init__(self, img, videogen):
            self.path = img
            self.videogen = videogen

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
            return len(self.videogen)

    dataloader = GetLoader(imgs, videogen)
    datas = DataLoaderX(dataloader, batch_size=args.batch_size, shuffle=False, drop_last=False,
                       num_workers=0)
    cal_count = 0
    if args.UHD:
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
    else:
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    # read_buffer = Queue(maxsize=10000)
    # _thread.start_new_thread(build_read_buffer, (imgs, read_buffer, videogen))
    # I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    # I1 = F.pad(I1, padding)
    # while True:
    #         frame = read_buffer.get()
    #         if frame is None:
    #             break
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
            cal_count = cal_count + 1
            I0 = I1
            I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
            I1 = F.pad(I1, padding)
            cnt += 1
            t1 = threading.Thread(imwrite('{}/{:0>9d}'.format(out,cnt),lastframe[:, :, ::-1]))
            t1.start()
            thread_list.append(t1)
            cnt += 1
            t2 = threading.Thread(imwrite('{}/{:0>9d}'.format(out,cnt),(((model.inference(I0, I1, args.UHD)[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w][:, :, ::-1]))
            if args.vector != 1:
                t2 = threading.Thread(imwrite('{}/{:0>9d}'.format(out,cnt),(((model.inference(I1, I0, args.UHD)[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w][:, :, ::-1]))
            t2.start()
            thread_list.append(t2)
            #cv2.imwrite('{}/{:0>9d}.png'.format(out,cnt),(((model.inference(I1, I0, args.UHD)[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))[:h, :w][:, :, ::-1]))
            lastframe = frame
    cnt += 1
    t3 = threading.Thread(imwrite('{}/{:0>9d}'.format(out,cnt),lastframe[:, :, ::-1]))
    t3.start()
    thread_list.append(t3)
    #cv2.imwrite('{}/{:0>9d}.png'.format(out,cnt), lastframe[:, :, ::-1])
    for t in thread_list:
        t.join()

t = json.load(open(args.json,'r',encoding='utf-8'))
tot_groups = len(t)
pbar = tqdm(total=tot_groups)

p = 0
while p != tot_groups:
    exp = 0
    while(exp != int(t[p]['exp'])):
        interpolate(t[p]['imgs'],args.output)
        for s in os.listdir(t[p]['imgs']):
            os.remove(os.path.join(t[p]['imgs'],s))
        fd = []
        for s in os.listdir(args.output):
            fd.append(os.path.join(args.output,s))
        fd.sort()
        nd = []
        for f in os.listdir(args.output):
            nd.append(f)
        nd.sort()
        l = len(nd)
        while(l>0):
            l = l - 1
            shutil.move(fd[l],os.path.join(t[p]['imgs'],nd[l]))
        exp += 1
    pbar.update(1)
    vgen = []
    for s in os.listdir(t[p]['imgs']):
        vgen.append(os.path.join(t[p]['imgs'],s))
    os.remove(vgen[0])
    os.remove(vgen[len(vgen)-1])
    vgen.remove(vgen[0])
    vgen.remove(vgen[len(vgen)-1])
    nts = 1 / (int(t[p]['need'])+1)
    ntsc = []
    ntp = 0
    while ntp < 1:
        ntp = ntp + nts
        if (ntp + 0.000001) < 1:
            ntsc.append(ntp)
    its = 1 / (len(vgen)+1)
    itsc = {}
    itp = 0
    n = 0
    while itp < 1:
        itp = itp +its
        if (itp + 0.000001) < 1:
            itsc[itp] = vgen[n]
            n += 1
    kpts = []
    for i in ntsc:
        min = 1
        kpt = ""
        for k in itsc:
            if abs(i-k) < min:
                min = abs(i-k)
                kpt = itsc[k]
        kpts.append(kpt)
    for s in vgen:
        if s not in kpts:
            os.remove(s)
    p = p + 1
