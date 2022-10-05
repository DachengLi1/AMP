#!/bin/bash

MY_IPADDR=$(hostname -i)
echo $MY_IPADDR
NNODE=$1
LOOP=`expr $NNODE - 1`

for i in `seq 0 $LOOP`
do
  echo "=> node $i"
  ssh -o StrictHostKeyChecking=no h$i.ray-dev8.BigLearning "screen -XS deep quit; screen -S deep; source ~/anaconda3/bin/activate; export http_proxy=http://ops:8888/; export https_proxy=http://ops:8888/; conda activate deep; cd /users/hzhang2/dacheng/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism; bash examples/pretrain_gpt2_distributed.sh $NNODE $i; exit";
done

