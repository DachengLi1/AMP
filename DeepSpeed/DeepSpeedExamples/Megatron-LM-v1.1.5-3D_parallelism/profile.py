import subprocess
import sys
import os
import json

sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/")
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")

world_size = 1
exp_name = "profile_alpha"
record_file = f"/users/dacheng2/amp_{exp_name}_data.txt"

if not os.path.exists(record_file):
    with open(record_file, 'w') as fp:
        pass
done = []

f = open(record_file, "r")
for line in f.readlines():
    done.append(line) 
f.close()

nlayer = 24
hidden = 1024
global_bs_list = [1,2,4, 8, 16]

model_conf = [(6,256),(12,512),(24,1024)]
#local_bs = 1
#B = 1

def factors(num):
    ret = []
    for i in range(1,num+1):
        if num % i == 0:
            ret.append(i)
    return ret

for local_bs in global_bs_list:
 #  local_bs = 1024
    file_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_profile.sh"
 
    gas = 1
    
    for (nlayer, hidden) in model_conf:
        print(done)
        conf = f" 1 1 {nlayer} {hidden} {local_bs} {gas} {exp_name}"
        record = f"mp: 1, dp: 1, pp: 1, nlayer: {nlayer}, hidden: {hidden}, global_bs: {local_bs}, local_bs: {local_bs}, gas: {gas} \n"
        
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
