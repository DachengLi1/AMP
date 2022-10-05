import time

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class test1(torch.nn.Module):
    def __init__(self):
        super(test1, self).__init__()
        self.weight = Parameter(torch.empty(
                    4096, 1024,dtype=torch.float16,
                    device=torch.cuda.current_device()))

    def forward(self, x):
        return F.linear(x, self.weight, None)

x = torch.randn(1024, 1024).to(torch.float16).to(0)
test = test1().to(0)
torch.cuda.synchronize()
time_s = time.time()
for i in range(1):
    test(x)
    if i == 0:
        print(test(x).shape)
torch.cuda.synchronize()
time_s = time.time()
for i in range(100):
    test(x)
    if i == 0:
        print(test(x).shape)
torch.cuda.synchronize()
TFLOPS = 100*2*1024*1024*4096 / 1e12
byte = 2*100 * 1024 * 4096 + 2*100*1024*1024
intensity = TFLOPS*1e12 / byte
load_time = byte / (3.2e11)
compute_time = TFLOPS / 65
print(f"use: {time.time() - time_s} with TFLOPs: {TFLOPS}, bytes: {byte}, intensity: {intensity}, load_time: {load_time}, compute_time: {compute_time}")
