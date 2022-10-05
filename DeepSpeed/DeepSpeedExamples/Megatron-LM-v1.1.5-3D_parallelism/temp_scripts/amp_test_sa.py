import math
import time
from collections import defaultdict
import operator
import random
import os

from tqdm import tqdm

import numpy as np

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import generate_initial, neighbor, cool_down
from cost import get_cost_c, get_cost_e, predict, rank_loss, AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import generate_ds_config, simulate, to_float_torch

# cluster information

# number of GPU per node, number of nodes
M = 1
N = 8

data_bs = 5000

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

#TODO: Find DGX-2 box config
cluster_info = {}

# inter-node bandwidth
for i in range(N):
    cluster_info[i] = torch.tensor([40]).float()

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([0.0512]).float(), 
                "sequence_length": torch.tensor([1.024]).float(), 
                "num_layers": torch.tensor([12]).float(), 
                "vocab_size":torch.tensor([0.52256]).float()}

config_h = int((10000 * model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())

exp_name = f"orca_8_{config_h}_{config_n}"

record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"
with open(record_file, "w") as fp:
    pass

# save this name to env
os.environ["amp_log_path"] = record_file

num_iter = 500
init_t = 1

global_bs = 8
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

model = AMP(model_config, exp_name)
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.9)

class candidate():
    def __init__(self, h, w, mbs, conf, partition, rank_map):
        self.h = h
        self.w = w
       # self.mbs = mbs
       # self.conf = conf
       # self.partiton = partition
       # self.rank_map = rank_map

    def __eq__(self, obj):
        equal = True
        for k, v in self.__dict__.items(): 
            if v != obj[k]:
                equal = False 
        return equal

    def __repr__(self):
        return self.__dict__.__repr__()

num_threads = 1
print("init")
h_w, micro_bs_list, configs, known = generate_initial(M, N, global_bs, threads=num_threads)
print("init done")

candidate_list = []


oth_list = [{"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w, "orig_pp": torch.ones(1,)*(M*N/(h*w))} for (h,w) in h_w]
args = (configs, [global_bs] * num_threads, micro_bs_list, cluster_info, model_config, oth_list)

cost_list = [ [] for i in range(num_threads)]

with torch.no_grad():
    rank_maps, partitions, costs = model(args)

for i in range(num_threads):
    candidate_list.append(candidate(h_w[i][0], h_w[i][1], micro_bs_list[i], configs[i], partitions[i], rank_maps[i]))

for kk in range(num_threads):
    with open(record_file, "a") as fp:
        fp.write(f"sa init (thread {kk}) exploring: {rank_maps[kk]}, {partitions[kk]}, {micro_bs_list[kk]}, {h_w[kk]} \n")
        print(f"sa init (thread {kk}) exploring: {rank_maps[kk]}, {partitions[kk]}, {micro_bs_list[kk]}, {h_w[kk]} \n")

for j in range(num_threads):
    cost_list[j].append(costs[j])

stop_index = []

# Track how good our cost model learns
loss_list = []

# Track how efficient our sa is, given cost model is good
cost_list = [ [] for i in range(num_threads)]

update_configs = []
update_h_w = []
update_micro_bs_list = []
update_oth_list = []

ready_data_count = 0

for i in tqdm(range(num_iter)):
    iter_s = time.time()
    cur_t = cool_down(i, num_iter, init_t)   
    
    new_configs = []
    new_h_w = []
    new_micro_bs_list = []
    new_oth_list = []
 
    for j in range(num_threads):
        if j in stop_index:
            continue
            
        h, w = h_w[j]
        mbs = micro_bs_list[j]

        step = neighbor((h, w, mbs), global_bs, known, M, N)
        if step is None:
            stop_index.append(j)
            with open(record_file, "a") as fp:
                fp.write(f"{j} has stopped with {configs[j]}\n")
                print(f"{j} has stopped with {configs[j]}\n")
            continue
        else:
            new_h, new_w, mbs, new_config = step
            
            new_oth = {"orig_mp": torch.ones(1,)*new_h, "orig_dp": torch.ones(1,)*new_w,
                       "orig_pp": torch.ones(1,)*(M*N/(new_h*new_w))}
            
            new_oth_list.append(new_oth)
            new_h_w.append((new_h, new_w))
            new_configs.append(new_config)
            new_micro_bs_list.append(mbs) 
  
            update_oth_list.append(new_oth)
            update_h_w.append((new_h, new_w))
            update_configs.append(new_config)
            update_micro_bs_list.append(mbs) 
            
            ready_data_count += 1
 
            if ready_data_count == data_bs:
                update_args = (update_configs, [global_bs] * len(update_configs), update_micro_bs_list, cluster_info, model_config, update_oth_list)
                update_rank_maps, update_partitions, update_costs = model(update_args)
                gt_costs = simulate(update_rank_maps, update_partitions, [torch.ones(1,)*global_bs]*len(update_configs), to_float_torch(update_micro_bs_list), model_config, update_oth_list, exp_name)
                
                loss = rank_loss(update_costs, gt_costs)
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

                with open(record_file, "a") as fp:
                    for kk in range(len(gt_costs)):
                        fp.write(f"Simulating at {i}: {update_rank_maps[kk]}, {update_partitions[kk]}, {update_micro_bs_list[kk]}, {update_oth_list[kk]}, with p_cost: {update_costs[kk]}, r_cost: {gt_costs[kk]} \n")                
                    fp.write(f"update at {i}: loss: {loss.item()} \n")                
                
                # update known infeasible micro-bs
                for kk in range(len(gt_costs)):
                    continue
                    if gt_costs[kk] == float("inf"):
                        inf_h = update_h_w[kk][0]
                        inf_w = update_h_w[kk][1]
                        inf_config = update_configs[kk]
                        inf_config_idx = known[(inf_h, inf_w)].index(inf_config)

                        limit = min(known[(inf_h, inf_w)][inf_config_idx][2], update_micro_bs_list[kk])
                        known[(inf_h, inf_w)][inf_config_idx][2] = limit
                        for ll in known[(inf_h, inf_w)][inf_config_idx][1]:
                            if ll > limit:
                                known[(inf_h, inf_w)][inf_config_idx][1].pop(known[(inf_h, inf_w)][inf_config_idx].index(ll))

                        fp.write("set mbs limit for ({inf_h},{inf_w}) to {limit}\n")                
                # update known infeasible micro-bs

                # clean up 
                ready_data_count = 0
                update_oth_list = []
                update_h_w = []
                update_configs = []
                update_micro_bs_list = []

    if len(stop_index) == num_threads:
        print(f"finish: {candidate_list}")
        break
        
    args = (new_configs, [global_bs] * num_threads, new_micro_bs_list, cluster_info, model_config, new_oth_list)
    
    
    with torch.no_grad():
        new_rank_maps, new_partitions, new_costs = model(args)
    
    for i in range(len(new_configs)):
        candidate_list.append(candidate(new_h_w[i][0], new_h_w[i][1], new_micro_bs_list[i], new_configs[i], new_partitions[i], new_rank_maps[i]))
    
    acc_prob = np.exp(np.minimum((costs - new_costs)/ (cur_t+1e-5) , 0))
    
    acc_index = (np.random.random(len(acc_prob)) < np.asarray(acc_prob))
    
    print(costs, new_costs, acc_prob, acc_index)  
    with open(record_file, "a") as fp:
        active_count = 0
        for kk in range(num_threads):
            if kk in stop_index:
                continue
            
            fp.write(f"sa at iteration {i} (thread {active_count}) exploring: {new_rank_maps[active_count]}, {new_partitions[active_count]}, {new_micro_bs_list[active_count]}, {new_oth_list[active_count]} \n")
            
            active_count += 1
    
    for j in range(num_threads):
        if j in stop_index:
            continue
            
    #    if acc_index[j]:
        if True:
            configs[j] = new_configs[j]
            costs[j] = new_costs[j]
            partitions[j] = new_partitions[j]
            rank_maps[j] = new_rank_maps[j]
            h_w[j] = new_h_w[j]
            micro_bs_list[j] = new_micro_bs_list[j]
        
        # booktracking cost for printing
        cost_list[j].append(costs[j])
    
    iter_used = time.time() - iter_s
    print(f"iter used {iter_used}")
    
