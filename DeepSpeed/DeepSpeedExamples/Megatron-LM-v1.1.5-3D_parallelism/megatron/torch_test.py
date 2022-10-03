import torch
import time

for i in range(10000000000000):
    a = torch.ones((1024,)).cuda()
    a = a**2
    time.sleep(1)
