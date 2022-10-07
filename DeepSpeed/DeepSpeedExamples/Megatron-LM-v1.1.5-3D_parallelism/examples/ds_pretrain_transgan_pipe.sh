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
mp_size=$1
# DeepSpeed Pipeline parallelism
pp_size=$2

NLAYERS=$3
NHIDDEN=$4
BATCHSIZE=$5
gas=$6
exp_name=$7

LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --gas ${gas} \
        --exp_name ${exp_name} \
        --num-layers ${NLAYERS} \
        --hidden-size ${NHIDDEN} \
        --num-attention-heads 1 \
        --seq-length 1024 \
	--fp16 \
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
        --save-interval 50000 \
        --eval-interval 100000 \
        --eval-iters 10000 \
        --tensorboard-dir ${LOGDIR}
        -gen_bs 16 \
        -dis_bs 16 \
        --accumulated_times 4 \
        --g_accumulated_times 8 \
        --dist-url 'tcp://localhost:10641' \
        --dist-backend 'nccl' \
        --dataset church \
        --data_path ./lsun \
        --bottom_width 9 \
        --img_size 256 \
        --max_iter 500000 \
        --gen_model ViT_custom_local544444_256_rp_noise \
        --dis_model ViT_scale3_local_new_rp \
        --g_window_size 18 \
        --d_window_size 12 \
        --g_norm pn \
        --df_dim 384 \
        --d_depth 3 \
        --g_depth 12,0,0,0,0,12 \
        --latent_dim 512 \
        --gf_dim ${NHIDDEN} \
        --num_workers 8 \
        --g_lr 0.0001 \
        --d_lr 0.0001 \
        --optimizer adam \
        --loss wgangp-eps \
        --wd 1e-3 \
        --beta1 0 \
        --beta2 0.99 \
        --phi 1 \
        --eval_batch_size 10 \
        --num_eval_imgs 50000 \
        --init_type xavier_uniform \
        --n_critic 4 \
        --val_freq 5000 \
        --print_freq 50 \
        --grow_steps 0 0 \
        --fade_in 0 \
        --patch_size 4 \
        --diff_aug translation,erase_ratio,color \
        --fid_stat fid_stat/fid_stats_church_256.npz \
        --ema 0.995 \
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
run_cmd="deepspeed --hostfile=/home/ubuntu/hostfile --master_port 9005 pretrain_transgan.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
