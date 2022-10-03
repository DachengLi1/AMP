import torch
import torch.distributed as dist
import time
import os
import sys

time_s = time.time()
s = 256
A = torch.randn((s//8,s,s * 2)).cuda()
B = torch.randn((s//8,s * 2,s)).cuda()

print(time.time() - time_s)

time_compute = 0
time_start = time.time()

for i in range(100):
    C = torch.bmm(A, B)

time_end = time.time()
time_compute = time_end - time_start

print(time_compute)

os.environ['MASTER_ADDR'] = '10.117.1.37'
os.environ['MASTER_PORT'] = '29500'
rank = int(sys.argv[1]) 
dist.init_process_group("nccl", rank=rank, world_size=2)

time_comm = 0
time_start = time.time()

for i in range(100):
    if rank == 0:
        dist.send(A, dst=1)
    else:
        dist.recv(A, src=0)

time_end = time.time()
time_comm = time_end - time_start

print(time_comm)

comp_unit = time_compute / (s // 8 * s * s * 2 * s)
comm_unit = time_comm / (s // 8 * s * s * 2 / 40)
print(f"alpha: {comp_unit / comm_unit}")

group = dist.new_group([0, 1])
time_comm = 0
time_start = time.time()

for i in range(100):
    dist.all_reduce(A, op=dist.ReduceOp.SUM, group=group)

time_end = time.time()
time_comm = time_end - time_start

print(time_comm / (s//8 * s * s * 2))

