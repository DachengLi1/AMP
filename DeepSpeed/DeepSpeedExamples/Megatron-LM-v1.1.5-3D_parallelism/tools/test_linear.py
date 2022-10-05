import time

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class test1(torch.nn.Module):
    def __init__(self):
        super(test1, self).__init__()
        self.weight = Parameter(torch.empty(
                    128, 32,dtype=torch.float16,
                    device=torch.cuda.current_device()))

    def forward(self, x):
        return F.linear(x, self.weight, None)

class test2(torch.nn.Module):
    def __init__(self):
        super(test2, self).__init__()
        self.weight = Parameter(torch.empty(
                    32, 128,dtype=torch.float16,
                    device=torch.cuda.current_device()))

    def forward(self, x):
        return F.linear(x, self.weight, None)


#print(f"TFlops: {2*32*128*32}")
x = torch.randn(128, 32).to(torch.float16).to(0)
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
TFLOPS = 100*2*32*128*128 / 1000000000000
byte = 2*100 * 128 * 32 + 2*100*128*32
intensity = TFLOPS*1e12 / byte
load_time = byte / (3.2e11)
compute_time = TFLOPS / 65
print(f"use: {time.time() - time_s} with TFLOPs: {TFLOPS}, bytes: {byte}, intensity: {intensity}, load_time: {load_time}, compute_time: {compute_time}")
#print(f"intensity: {100*2*32*128*128*32 / 1000000000000}/
"""
x = torch.randn(32, 32, 128).to(torch.float16).to(0)
test = test2().to(0)

torch.cuda.synchronize()
time_s = time.time()
for i in range(100):
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
TFLOPS = 100*2*32*128*32*32 / 1000000000000
byte = 2*100 * 128 * 32 + 2*100*32*32*128
intensity = TFLOPS / byte
load_time = byte / (3.2e11)
compute_time = TFLOPS / 65
print(f"use: {time.time() - time_s} with TFLOPs: {TFLOPS}, bytes: {byte}, intensity: {intensity}, load_time: {load_time}, compute_time: {compute_time}")"""
