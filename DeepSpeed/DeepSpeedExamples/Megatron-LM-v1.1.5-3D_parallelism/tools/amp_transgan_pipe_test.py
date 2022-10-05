import math
import time
from collections import defaultdict
import operator
import random
import os
import copy

from tqdm import tqdm

import numpy as np

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import generate_initial, neighbor, cool_down, candidate
from cost import get_cost_c, get_cost_e, rank_loss, AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import simulate, to_float_torch

# cluster information

# number of GPU per node, number of nodes
M = 4
N = 4

data_bs = 3

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

#TODO: Find DGX-2 box config
cluster_info = {}

# inter-node bandwidth, intra-node bandwidth
for i in range(N):
    cluster_info[i] = [torch.tensor([10 * 1e9 / 32]).float(), torch.tensor([170 * 1e9 / 32]).float()]

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([1024]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"transgan",
                "depth": [5, 4, 4, 4, 4, 4],
                "bottom":10}
print(model_config)
config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"aws_4_4_{config_h}_{config_n}_sa"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# save this name to env
os.environ["amp_log_path"] = record_file

num_iter = 500
init_t = 10

global_bs = 32
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

model = AMP(model_config, exp_name)
num_threads = 1

#h_w, micro_bs_list, configs, known = generate_initial(M, N, global_bs, threads=num_threads)

h_w = [(1,4)]
micro_bs_list = [1]

configs = [  np.asarray([
             np.asarray([1, 2, 3, 4]),
             np.asarray([1, 2, 3, 4]),
             np.asarray([1, 2, 3, 4]),
             np.asarray([1, 2, 3, 4])  ])   ]

oth_list = [{"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w, "orig_pp": torch.ones(1,)*(M*N/(h*w))} for (h,w) in h_w]
args = (configs, [global_bs] * num_threads, micro_bs_list, cluster_info, model_config, oth_list)


with torch.no_grad():
    rank_maps, partitions, costs = model(args, model_type="transgan")

print(costs)

gt_cost = simulate(rank_maps, partitions, torch.ones(1,)*global_bs, to_float_torch(micro_bs_list), model_config, oth_list, exp_name)
    #gt_cost = [np.random.randn()]
gt_cost = gt_cost[0]
print(gt_cost)
#with open(record_file, "a") as fp:
#    fp.write(f"Simulating after training: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {want_simulate[i][1]}, r_cost: {gt_cost} \n")
