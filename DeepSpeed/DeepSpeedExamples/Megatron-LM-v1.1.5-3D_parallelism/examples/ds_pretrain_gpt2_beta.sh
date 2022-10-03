#! /bin/bash

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6009

DATA_PATH=/home/ubuntu/datasets/datasets/preprocessed_data/my-gpt2_text_document
VOCAB_PATH=/home/ubuntu/datasets/datasets/gpt2-vocab.json
MERGE_PATH=/home/ubuntu/datasets/datasets/gpt2-merges.txt
CHECKPOINT_PATH=/home/ubuntu/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ckpts/gpt2_1542m_ds_distributed

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
#config_json="$script_dir/ds_zero_stage_2_config.json"
config_json="$script_dir/ds_config.json"

# Megatron Model Parallelism
mp_size=2
# DeepSpeed Pipeline parallelism
pp_size=1
dp_size=1
NLAYERS=$1
NHIDDEN=$2
BATCHSIZE=1
gas=1

exp_name=$3

WORLD_SIZE=$(($pp_size*$dp_size*$mp_size))
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --gas ${gas} \
        --exp_name ${exp_name} \
        --num-layers ${NLAYERS} \
        --hidden-size ${NHIDDEN} \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 1.5e-4 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --log-interval 20 \
        --fp16 \
        --save-interval 50000 \
        --eval-interval 100000 \
        --eval-iters 10000 \
        --tensorboard-dir ${LOGDIR}
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
            "

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

# %@
run_cmd="deepspeed --hostfile=/home/ubuntu/hostfile --master_port 9005 --num_nodes 2 --num_gpus 1 pretrain_gpt2.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
