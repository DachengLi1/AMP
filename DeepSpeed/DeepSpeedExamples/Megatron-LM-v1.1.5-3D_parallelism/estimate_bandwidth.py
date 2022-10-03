import torch
import torch.distributed as dist
import sys
import os

init_method = 'tcp://'
master_ip = "172.31.46.201"
master_port = os.getenv('MASTER_PORT', '6001')
init_method += master_ip + ':' + master_port

rank = int(sys.argv[1])
torch.cuda.set_device(rank)

torch.distributed.init_process_group(
    backend="nccl",
    world_size=2, rank=rank,
    init_method=init_method)

print("finish init")

torch.manual_seed(0)
os.environ["NCCL_ALGO"] = "^tree,collnet"
A = torch.randn((1024, 1024)).to(rank)
import time
time_s = time.time()
for i in range(int(sys.argv[2])):
    #dist.all_reduce(A)
    dist.broadcast(A, src=0)
print(f"used time {time.time() - time_s}")
