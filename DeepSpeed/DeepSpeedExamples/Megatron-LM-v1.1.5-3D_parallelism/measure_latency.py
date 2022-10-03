import torch
import time

a = torch.randn(128, 1000, 64).cuda()

time_s = time.time()
a = a ** 2
time_e = time.time()
torch.cuda.synchronize()
time_sync = time.time()

print(time_e - time_s, time_sync - time_e)
