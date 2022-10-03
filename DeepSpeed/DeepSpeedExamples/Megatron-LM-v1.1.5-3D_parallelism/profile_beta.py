import subprocess
import sys
import os
import json

sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/")
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")

exp_name = "profile_beta"
record_file = f"/users/dacheng2/amp_{exp_name}_data.txt"

if not os.path.exists(record_file):
    with open(record_file, 'w') as fp:
        pass
done = []

f = open(record_file, "r")
for line in f.readlines():
    done.append(line) 
f.close()

global_bs_list = [1, 2, 4, 8, 16]

model_conf = [(3,128), (6,256),(12,512)]
#local_bs = 1
#B = 1

def factors(num):
    ret = []
    for i in range(1,num+1):
        if num % i == 0:
            ret.append(i)
    return ret


for (nlayer, hidden) in model_conf:
 #  local_bs = 1024
    file_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_profile_beta.sh"
 
    gas = 1
    
    for global_bs in global_bs_list:
        for num_worker in [1,2]:
            local_bs = global_bs
            conf = f" {num_worker} 1 1 {nlayer} {hidden} {local_bs} {gas} {exp_name}"
            record = f"mp: {num_worker}, dp: 1, pp: 1, nlayer: {nlayer}, hidden: {hidden}, global_bs: {global_bs}, local_bs: {local_bs}, gas: {gas} \n"
        
            if record in done:
                continue
                
             # Chenge the ds_config
            json_path = '/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_config.json'
            with open(json_path) as json_file:
                dict = json.load(json_file)
                dict["train_micro_batch_size_per_gpu"] = local_bs
                dict["gradient_accumulation_steps"] = gas

            with open(json_path, 'w') as outfile:
                json.dump(dict, outfile)                               
         
            cmd = "bash " + file_path + conf
                
            f = open(record_file, "a")
                
            f.write(record)
            f.close()
                            
            process = subprocess.run(cmd, shell=True)
