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
    cluster_info[i] = [torch.tensor([50]).float(), torch.tensor([75]).float()]

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([0.1024]).float(), 
                "sequence_length": torch.tensor([1.024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([0.52256]).float()}

config_h = int((10000 * model_config["hidden_size"]).item())
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
optimizer = optim.SGD(model.parameters(), lr=.1)
with open(record_file, "w") as fp:
    fp.write(f"init param to: ({model.alpha}, {model.beta})")

num_threads = 1

budget = 30
update_budget = 5
remain_update_budget = update_budget

h_w, micro_bs_list, configs, known = generate_initial(M, N, global_bs, threads=num_threads)

oth_list = [{"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w, "orig_pp": torch.ones(1,)*(M*N/(h*w))} for (h,w) in h_w]
args = (configs, [global_bs] * num_threads, micro_bs_list, cluster_info, model_config, oth_list)

cost_list = [ [] for i in range(num_threads)]

with torch.no_grad():
    rank_maps, partitions, costs = model(args)

for kk in range(num_threads):
    with open(record_file, "a") as fp:
        #fp.write(f"sa init (thread {kk}) exploring: {partitions[kk]}, {micro_bs_list[kk]}, {h_w[kk]} \n")
        fp.write(f"sa init (thread {kk}) exploring: {rank_maps[kk]}, {partitions[kk]}, {micro_bs_list[kk]}, {h_w[kk]} \n")

for j in range(num_threads):
    cost_list[j].append(costs[j])

stop_index = []

# Track how good our cost model learns
loss_list = []

# Track how efficient our sa is, given cost model is good
cost_list = [ [] for i in range(num_threads)]

update_configs = copy.deepcopy(configs)
update_h_w = copy.deepcopy(h_w)
update_micro_bs_list = copy.deepcopy(micro_bs_list)
update_oth_list = copy.deepcopy(oth_list)
update_candidates = [candidate(h_w[i][0], h_w[i][1], micro_bs_list[i], configs[i], partitions[i], rank_maps[i]) for i in range(len(configs))]

ready_data_count = 0
simulated_settings = []
want_simulate = [(candidate(h_w[i][0], h_w[i][1], micro_bs_list[i], configs[i], partitions[i], rank_maps[i]), costs[i]) for i in range(len(configs))]

for i in tqdm(range(num_iter)):
    iter_s = time.time()
    cur_t = cool_down(i, num_iter, init_t)   
    
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
                fp.write(f"{costs}, {new_cost}, {acc_prob}, {accept} \n")
            
            # want to simulate but no budget, possibly simulate at last
            want_simulate.append((new_candidate, new_cost))
            
            # If we accept, update current
            if accept:
                cost_list[j].append(new_cost)
                with open(record_file, "a") as fp:
                    fp.write(f"(thread {j}) accepts: {new_rank_map}, {new_partition}, {new_mbs}, {new_oth} with p_cost: {new_cost}\n")
                
                configs[j] = new_config
                costs[j] = new_cost
                partitions[j] = new_partition
                rank_maps[j] = new_rank_map
                h_w[j] = (new_h, new_w)
                micro_bs_list[j] = new_mbs
                
                simulated_list = [k for (k, v) in simulated_settings]
                if new_candidate not in simulated_list:
                    update_oth_list.append(new_oth)
                    update_h_w.append((new_h, new_w))
                    update_configs.append(new_config)
                    update_micro_bs_list.append(new_mbs)
                    update_candidates.append(new_candidate)
                    simulated_settings.append((new_candidate, new_cost)) 
                    ready_data_count += 1
                    with open(record_file, "a") as fp:
                        fp.write(f"adding {new_candidate} to simulated. \n")                
                else: 
                    with open(record_file, "a") as fp:
                        fp.write(f"{new_candidate} has been simulated. \n")                
 
                if remain_update_budget > 0:
                    if ready_data_count >= data_bs:
                        update_args = (update_configs, [global_bs] * len(update_configs), update_micro_bs_list, cluster_info, model_config, update_oth_list)
                        update_rank_maps, update_partitions, update_costs = model(update_args)
                        gt_costs = simulate(update_rank_maps, update_partitions, torch.ones(1,)*global_bs, to_float_torch(update_micro_bs_list), model_config, update_oth_list, exp_name)
                        #gt_costs = np.random.randn(len(update_rank_maps))
                        #for gt in range(len(gt_costs)):
                        #    if gt_costs[gt] > 0:
                        #        gt_costs[gt] = float("inf")

                        remain_update_budget -= len(gt_costs)
                    
                        loss = rank_loss(update_costs, gt_costs)
                        loss.backward()
                        loss_list.append(loss.item())
                        optimizer.step()
                        optimizer.zero_grad()
                        with open(record_file, "a") as fp:
                            for kk in range(len(gt_costs)):
                                fp.write(f"Simulating at {i}: {update_rank_maps[kk]}, {update_partitions[kk]}, {update_micro_bs_list[kk]}, {update_oth_list[kk]}, with p_cost: {update_costs[kk]}, r_cost: {gt_costs[kk]} \n")                
                            fp.write(f"update at {i}: loss: {loss.item()} \n")                
                            fp.write(f"update param to: ({model.alpha}, {model.beta})")
                        
                        # update known infeasible micro-bs
                        for kk in range(len(gt_costs)):
                            if gt_costs[kk] == float("inf"):
                                inf_h = update_h_w[kk][0]
                                inf_w = update_h_w[kk][1]
                                inf_config = update_configs[kk]
                                inf_mbs = update_micro_bs_list[kk]
                                
                                inf_config_idx = None
                                for find_idx in range(len(known[(inf_h, inf_w)])):
                                    if (known[(inf_h, inf_w)][find_idx][0] == inf_config).all():
                                        inf_config_idx = find_idx
                                if inf_config_idx is not None:
                                    for mbs_ in known[(inf_h, inf_w)][inf_config_idx][1]:
                                        if mbs_ > inf_mbs:
                                            _mbs_idx = known[(inf_h, inf_w)][inf_config_idx][1].index(mbs_)
                                            known[(inf_h, inf_w)][inf_config_idx][1].pop(_mbs_idx)
                                            if len(known[(inf_h, inf_w)][inf_config_idx][1]) == 0:
                                                known[(inf_h, inf_w)].pop(inf_config_idx)
                                            with open(record_file, "a") as fp:
                                                fp.write("set mbs limit for ({inf_h},{inf_w},{inf_config}) to {inf_mbs}\n")                
                        # clean up 
                        ready_data_count = 0
                        update_oth_list = []
                        update_h_w = []
                        update_configs = []
                        update_micro_bs_list = []
                        for find_candidate in update_candidates:
                            oth_idx = None
                            
                            for find_candidate_oth in want_simulate:
                                if find_candidate == find_candidate_oth[0]:
                                    find_candidate_oth_idx = want_simulate.index(find_candidate_oth)
                                    want_simulate.pop(find_candidate_oth_idx)
                                    oth_idx = find_candidate_oth_idx
                                    print("removed from want simulate")
                            if oth_idx is None:
                                print("did not find in want simulate")

            else:
                with open(record_file, "a") as fp:
                    fp.write(f"thread {j} rejects candidate: {new_h, new_w} {new_mbs} with p_cost: {new_cost}\n")
            

    if len(stop_index) == num_threads:
        break

    i += 1
        
# sorted simulated settings
sorted_settings = sorted(simulated_settings, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    fp.write(f"All threads finished.\n")
#    for item in sorted_settings:
#        fp.write(f"{item}")

want_simulate = sorted(want_simulate, key = lambda kv: kv[1])
for i in range(budget - update_budget - (update_budget - remain_update_budget)):
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
        fp.write(f"Simulating after training: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {want_simulate[i][1]}, r_cost: {gt_cost} \n")




