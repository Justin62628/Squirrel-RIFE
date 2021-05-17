import numpy
import torch

if  torch.cuda.is_available():
    n = 0
    while n != torch.cuda.device_count():
        print(torch.cuda.get_device_name(n))
        n += 1