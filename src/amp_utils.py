import torch
import os
import subprocess
import json
import torch
import spur

# returns the rank to axis. If pp_deg=dp_deg=mp_deg=2, rank 3 gives (0,1,1).
# This is deepspeed method
def rank2axis(rank, mp_deg, dp_deg, pp_deg):
    pp = rank // (mp_deg * dp_deg)
    remainder = rank % (mp_deg * dp_deg)

    dp = remainder // (mp_deg)
    remainder = remainder % mp_deg

    mp = remainder

    return (pp, dp, mp)

# returns the axis to rank. If pp_deg=dp_deg=mp_deg=2, (0,1,1) gives 3
def axis2rank(axis, mp_deg, dp_deg, pp_deg):
    pp, dp, mp = axis
    return mp + mp_deg * dp + (mp_deg * dp_deg) * pp

def factor(N, upper=None):
    if upper is None:
        upper = N
    ret = []
    for i in range(1, upper+1):
        if N % i == 0:
            ret.append(i)
    return ret

#def generate_ds_config(M, N, mp_deg, dp_deg, pp_deg):
#    ds_config = torch.zeros(M, N)
#    config = []
#    cur_index = 1
#    for i in range(int(pp_deg.item())):
#        config.extend([cur_index] * (int(mp_deg.item()) * int(dp_deg.item())))
#        cur_index += 1
#
#    for i in range(M):
#        for j in range(N):
#            ds_config[i][j] = config[i * N + j]    
#    
#    return ds_config

def get_host():
    home = os.environ['HOME']
    ret = []
    with open(os.path.join(home, "hostfile"), "r") as f:
        lines = f.readlines()
        for line in lines:
            ret.append(line.split(" ")[0])
    return ret

def remove_remote(hostname, path):
    shell = spur.SshShell(hostname=hostname, username="ubuntu", private_key_file="/home/ubuntu/.ssh/id_rsa")
    result = shell.run(["rm", "-rf", path])

#def simulate(candidate_list, gbs, model_config, exp_name):
def simulate(rank_maps, partitions, gbs, micro_bs_list, model_config, oth_list, exp_name):
    home_path = os.environ['HOME']
    exp_name = "simulate_" + exp_name
    gt_costs = []
    h = float(model_config["hidden_size"].item())
    n = float(model_config["num_layers"].item()) 
    model_type = model_config["type"]

    config_h = int((model_config["hidden_size"]).item())
    config_n = int(n)

    hosts = get_host()[1:]
    for i in range(len(rank_maps)):
        rank_map = rank_maps[i]
        print(f"debug rank_map {rank_map}")
        partition = partitions[i]
        micro_bs = int(micro_bs_list[i].item())
        oth = oth_list[i]
    
        mp = int(oth["mp_deg"].item())
        pp = int(oth["pp_deg"].item())
        dp = int(oth["dp_deg"].item())
        gas =  int((gbs / (dp * micro_bs)).item())
        dir_path = os.path.join(home_path, 'tmp')

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        rmap_path = os.path.join(dir_path, f"rank_map.json")
        if os.path.exists(rmap_path):
            os.remove(rmap_path)
        
        # We want to do it ourself
        if rank_map is not None:
            with open(rmap_path, 'w') as outfile:
                json.dump(rank_map, outfile)
        
            for oth_host in hosts:
                print(f"Transferring rmap to {oth_host} all other hosts:{hosts}")
                subprocess.run(f"scp {rmap_path} ubuntu@{oth_host}:{rmap_path}", shell=True)
        else:
            for oth_host in hosts:
                remove_remote(oth_host, rmap_path)
        partition_path = os.path.join(dir_path, f"partition.json")
        if os.path.exists(partition_path):
            os.remove(partition_path)
        if partition is not None:
            with open(partition_path, 'w') as outfile:
                json.dump(partition, outfile)
            for oth_host in hosts:
                subprocess.run(f"scp {partition_path} ubuntu@{oth_host}:{partition_path}", shell=True)
        else:
            for oth_host in hosts:
                remove_remote(oth_host, partition_path)

        json_path = os.path.join(home_path, 'AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_config.json')

        if model_type == "gpt2":
            script_path = os.path.join(home_path,"AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh")
            conf = f" {mp} {pp} {config_n} {config_h} {micro_bs} {gas} {exp_name}"
        else:
            assert model_type == "transgan"
            script_path = os.path.join(home_path,"AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_transgan_pipe.sh")
            conf = f" {mp} {pp} {config_n} {config_h} {micro_bs} {gas} {exp_name}"

        dir_path = os.path.join(home_path, "amp_simulate")
        assert os.path.isdir(dir_path)
        record_path = os.path.join(dir_path, f"{exp_name}.txt")
        if os.path.exists(record_path):
            os.remove(record_path)
        with open(record_path, "w") as tf:
            pass
        
        
        if partition is not None:
            assert len(partition) - 1 == pp ,"bug in pipeline amp"
        with open(json_path) as json_file:
            dict = json.load(json_file)
            dict["train_micro_batch_size_per_gpu"] = micro_bs
            dict["gradient_accumulation_steps"] = gas

        with open(json_path, 'w') as outfile:
            json.dump(dict, outfile)
        cmd = "bash " + script_path + conf
        print(cmd)
        for oth_host in hosts:
            subprocess.run(f"scp {json_path} ubuntu@{oth_host}:{json_path}", shell=True)
        subprocess.run(cmd, shell=True)
        
        with open(record_path, "r") as f:
            lines = f.readlines()
            if len(lines) == 1:
                gt_costs.append(float(lines[0].rstrip()))
            else:
                print("--------------- cannot run--------------------")
                # Can not run for some reason
                assert len(lines) == 0
                gt_costs.append(float("inf"))        
        # delete all the tmp file created
        if os.path.exists(rmap_path):
            os.remove(rmap_path)
        if os.path.exists(partition_path):
            os.remove(partition_path) 
        if os.path.exists(record_path):
            os.remove(record_path)

    return gt_costs

def to_float_torch(int_list):
    ret = []
    for i in int_list:
        ret.append((torch.ones(1,) * i).float())
    return ret
