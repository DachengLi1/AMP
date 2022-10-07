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

from cost import get_cost_c, get_cost_e,  AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import simulate, to_float_torch

# cluster information

# number of GPU per node, number of nodes
M = 4
N = 4

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# inter-node bandwidth, intra-node bandwidth
for i in range(N):
    cluster_info[i] = [torch.tensor([50 * 1e9 / 32]).float(), torch.tensor([50 * 1e9 / 32]).float()]

model_config = {"hidden_size": torch.tensor([1024]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"gpt2"}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"homogeneous_sa"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# save this name to env
os.environ["amp_log_path"] = record_file

num_iter = 100
init_t = 10
global_bs = 32

assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

model = AMP(model_config, exp_name)
with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")                
    fp.write(f"gbs:{global_bs}\n")

# Generate a single initialization point
h_w, micro_bs, config, known = generate_initial(M, N, global_bs, threads=1)
h,w = h_w

oth = {"mp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(M*N/(h*w))}
sa_args = (config, global_bs, micro_bs, cluster_info, model_config, oth)

with torch.no_grad():
    rank_map, partition, cost = model(args)

with open(record_file, "a") as fp:
    fp.write(f"sa init with: {rank_map}, {partition}, {micro_bs}, {h_w} \n")

want_simulate = [(candidate(h, w, micro_bs, config, partition, rank_map), cost)]

# SA algorithm with num_iter iterations
for i in tqdm(range(num_iter)):
    iter_s = time.time()
    cur_t = cool_down(i, num_iter, init_t)   
    
    h, w = h_w[j]
    mbs = micro_bs_list[j]

    step = neighbor((h, w, mbs), global_bs, known, M, N)
    if step is None:
        stop_index.append(j)
        with open(record_file, "a") as fp:
            fp.write(f"{j} has stopped with {configs[j]}\n")
        continue
    else:
            new_h, new_w, new_mbs, new_config = step
            
            new_oth = {"orig_mp": torch.ones(1,)*new_h, "orig_dp": torch.ones(1,)*new_w,
                       "orig_pp": torch.ones(1,)*(M*N/(new_h*new_w))}
            
            # Check whether this has been simulated
            new_args = (new_config, global_bs, new_mbs, cluster_info, model_config, new_oth)
            with torch.no_grad():
                new_rank_map, new_partition, new_cost = model(new_args)

            new_candidate = candidate(new_h, new_w, new_mbs, new_config, new_partition, new_rank_map)  
            
            acc_prob = np.exp(np.minimum((costs[j] - new_cost)/ (cur_t+1e-5) , 0))
    
            dice = np.random.random(1,)[0]
            
            accept = dice < acc_prob
            with open(record_file, "a") as fp:
                    fp.write(f"(thread {j}) predicts: {new_rank_map}, {new_partition}, {new_mbs}, {new_oth} with p_cost: {new_cost}\n")
                    fp.write(f"{costs}, {new_cost}, {acc_prob}, {accept} \n")
                    
            # If we accept, update current
            if accept:
                with open(record_file, "a") as fp:
                    #fp.write(f"(thread {j}) accepts: {new_rank_map}, {new_partition}, {new_mbs}, {new_oth} with p_cost: {new_cost}\n")
                    fp.write("\n")
                configs[j] = new_config
                costs[j] = new_cost
                partitions[j] = new_partition
                rank_maps[j] = new_rank_map
                h_w[j] = (new_h, new_w)
                micro_bs_list[j] = new_mbs
                
                simulate_list = [k for (k, v) in want_simulate]
                if new_candidate not in simulate_list:
                    want_simulate.append((new_candidate, new_cost))               
                else: 
                    with open(record_file, "a") as fp:
                        fp.write(f"{new_candidate} has been added. \n")                
            else:
                with open(record_file, "a") as fp:
                    #fp.write(f"thread {j} rejects candidate: {new_h, new_w} {new_mbs} with p_cost: {new_cost}\n")
                    fp.write("\n")

    if len(stop_index) == num_threads:
        break

    i += 1
        
# sorted simulated settings
with open(record_file, "a") as fp:
    fp.write(f"All threads finished.\n")
    fp.write(f" ---------- printing sorted---------------- {len(want_simulate)}")
#    for item in sorted_settings:
#        fp.write(f"{item}")

want_simulate = sorted(want_simulate, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    fp.write(f"{want_simulate}")

for i in range(budget):
    can = want_simulate[i][0]
    rmap = can.rank_map
    partition = can.partition
    mbs = can.mbs
    h = can.h
    w = can.w  
    oth = [{"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w,
                               "orig_pp": torch.ones(1,)*(M*N/(h*w))}]
    gt_cost = simulate([rmap], [partition], torch.ones(1,)*global_bs, to_float_torch([mbs]), model_config, oth, exp_name)
    #gt_cost = [np.random.randn()]
    gt_cost = gt_cost[0]
    with open(record_file, "a") as fp:
        fp.write(f"Simulating after sa: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {want_simulate[i][1]}, r_cost: {gt_cost} \n")




