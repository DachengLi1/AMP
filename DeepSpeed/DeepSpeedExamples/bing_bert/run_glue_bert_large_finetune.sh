LOG_DIR="log"
if [ ! -d "$LOG_DIR" ]; then
  mkdir $LOG_DIR
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

NGPU=$1

echo "Started scripts"

TASK=$2
EFFECTIVE_BATCH_SIZE=$3
LR=$4
NUM_EPOCH=$5
base_dir=`pwd`
model_name="bert_large"
JOBNAME=$6
CHECKPOINT_PATH=$7
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${model_name}/${JOBNAME}_bsz${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}"

GLUE_DIR="/data/GlueData"

MAX_GPU_BATCH_SIZE=32
PER_GPU_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE/NGPU))
if [[ $PER_GPU_BATCH_SIZE -lt $MAX_GPU_BATCH_SIZE ]]; then
       GRAD_ACCUM_STEPS=1
else
       GRAD_ACCUM_STEPS=$((PER_GPU_BATCH_SIZE/MAX_GPU_BATCH_SIZE))
fi

echo "Fine Tuning $CHECKPOINT_PATH"
run_cmd="python3.6 -m torch.distributed.launch \
       --nproc_per_node=${NGPU} \
       --master_port=12346 \
       run_glue_classifier_bert_large.py \
       --task_name $TASK \
       --do_train \
       --do_eval \
       --deepspeed \
       --deepspeed_transformer_kernel \
       --fp16 \
       --preln \
       --deepspeed_config ${base_dir}/glue_bert_large.json \
       --do_lower_case \
       --data_dir $GLUE_DIR/$TASK/ \
       --bert_model bert-large-uncased \
       --max_seq_length 128 \
       --train_batch_size ${PER_GPU_BATCH_SIZE} \
       --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
       --learning_rate ${LR} \
       --num_train_epochs ${NUM_EPOCH} \
       --output_dir ${OUTPUT_DIR}_${TASK} \
       --model_file $CHECKPOINT_PATH &> $LOG_DIR/${model_name}/${JOBNAME}_${TASK}_bzs${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}_${NGPU}_deepspeed-kernel.txt
       "
echo ${run_cmd}
eval ${run_cmd}
