# 1. 设置 HF_HOME 为 /dev/shm/hf_home
mkdir -p /dev/shm/hf_home
export HF_HOME=/dev/shm/hf_home

export XLA_HLO_DEBUG=1
export MODEL=unsloth/Meta-Llama-3.1-70B-Instruct
export VLLM_TPU_PROFILE_DURATION_MS=2000
export VLLM_TPU_PROFILE_DELAY_MS=1000


python3 tpu_profiling_ttft.py \
    --model $MODEL \
    --input-len 5600 \
    --output-len 83000 \
    --batch-size 4 \
    --enforce-eager \
    --profile-result-dir profiles \
    --tensor-parallel-size 8 \
    --num-iters-warmup 2


export XLA_HLO_DEBUG=1
export MODEL=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
export VLLM_TPU_PROFILE_DURATION_MS=3000
export VLLM_TPU_PROFILE_DELAY_MS=0

python3 tpu_profiling_ttft.py \
    --model $MODEL \
    --input-len 5600 \
    --output-len 83000 \
    --batch-size 4 \
    --enforce-eager \
    --profile-result-dir profiles \
    --tensor-parallel-size 8 \
    --num-iters-warmup 2