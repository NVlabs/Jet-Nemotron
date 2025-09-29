#!/bin/bash

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export LM_HARNESS_CACHE_PATH=".cache/lm_harness_cache"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_ALLOW_CODE_EVAL=1

mkdir -p .cache/lm_harness_cache

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

MODEL_NAME_OR_PATH=${1-"jet-ai/Jet-Nemotron-4B"}

read -r -d '' cmd_prefix <<EOF
torchrun --nnodes 1 --nproc_per_node=$gpu_count --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
jetai/evaluation/meta_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir results/eval/Jet-Nemotron-4B
EOF

cmd_math_gen="${cmd_prefix} \
    --eval_config jetai/evaluation/configs/math_gen.yaml \
    --eval_batch_size 16 \
    --generation_num_chunks 32
"

cmd_math_mc="${cmd_prefix} \
    --eval_config jetai/evaluation/configs/math_mc.yaml \
    --eval_batch_size 16 \
"

cmds=(
    "$cmd_math_gen"
    "$cmd_math_mc"
)

for cmd in "${cmds[@]}"; do
    echo $cmd
    bash -c "${cmd}"
done