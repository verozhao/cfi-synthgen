#!/bin/bash
# usage: train_run_1024.sh <gpus, e.g. 1,2,3> <run_name> <dataset_root[,dataset_root2]> <max_train_steps> \
#                          <grad_accum> <ckpt_every> <init_lora> [port]
# 1024 px per view, one sample per GPU (DDP over the listed GPUs), 4-bit FLUX base, checkpointed
# activations in pinned CPU RAM. init_lora seeds checkpoint-0 (the released LoRA or one of ours).
set -u
source /mnt/nvme1n1/veronica_unitex/env.sh
GPUS=$1; NAME=$2; DATA=$3; STEPS=$4; ACC=$5; CKPT=$6; INIT=$7; PORT=${8:-29517}
NP=$(echo $GPUS | tr ',' '\n' | grep -c .)
FLUX=$(ls -d $R/cache/hf/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/ | head -1)
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CFI_ROOT=$R/repos/cfi-synthgen PYTHONPATH=$R/repos/cfi-synthgen
OUT=$R/runs/$NAME
mkdir -p $OUT/checkpoint-0
if [ ! -f $OUT/checkpoint-0/pytorch_lora_weights.safetensors ]; then
  cp $INIT $OUT/checkpoint-0/pytorch_lora_weights.safetensors
  echo "checkpoint-0 = $INIT" > $OUT/INIT_FROM.txt
fi
MP="--num_processes $NP"
[ $NP -gt 1 ] && MP="--multi_gpu $MP --main_process_port $PORT"
cd $R/repos/train/UniTEX-FLUX
CUDA_VISIBLE_DEVICES=$GPUS $R/venv/bin/accelerate launch $MP --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  launch.py \
  --pretrained_model_name_or_path $FLUX \
  --dataset_impl cfi --cfi_root $CFI_ROOT --cfi_skip_broken --dataset_name_list ${DATA//,/ } \
  --view_resolution 1024 --resolution 1024 6144 --n_rows 1 --n_cols 6 \
  --use_complex_dataset --six_views_or_four_views --dual_image --control_image --both_ccm_normal_condition \
  --mixed_precision bf16 --lora_rank 16 --lora_alpha 16 --optimizer prodigy --learning_rate 1.0 \
  --guidance_scale 1.0 --train_batch_size 1 --gradient_accumulation_steps $ACC --gradient_checkpointing \
  --random_drop_noise --random_drop_noise_probability 0.75 \
  --random_drop_condition --random_drop_condition_probability 0.25 \
  --max_train_steps $STEPS --checkpointing_steps $CKPT --validation_steps 10000000 --dataloader_num_workers 4 \
  --zero_text_embeds --quantize_base nf4 --offload_activations --pos_scale ${POS_SCALE:-1.0} \
  --glyph --glyph_config $R/data/glyph_train.json --text_keep_boost 0.3 \
  --resume_from_checkpoint latest --output_dir $OUT --report_to tensorboard --tasks texturing --seed 666
