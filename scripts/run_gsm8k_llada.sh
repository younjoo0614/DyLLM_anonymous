#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/path/to/LLaDA-8B-Instruct/"
TASKS="gsm8k"
BATCH_SIZE=16
NUM_SHOT=5
LIMIT=256
NUM_FULL_STEPS=4
THRESHOLDS=(0.99 0.985 0.98)
BLOCK_SIZE=(8 16 32)



for THRESHOLD in "${THRESHOLDS[@]}"; do
  LOG_FILE="${TASKS}_dyllm_llada_${THRESHOLD}_pruned.log"
  for MAX_NEW_TOKENS in 256; do
    for TPS in 1; do
      NUM_STEPS=$(( MAX_NEW_TOKENS / TPS ))

      echo "==================================================" | tee -a "${LOG_FILE}"
      echo "Running max_new_tokens=${MAX_NEW_TOKENS}, tokens_per_step=${TPS}, num_steps=${NUM_STEPS}" | tee -a "${LOG_FILE}"
      echo "==================================================" | tee -a "${LOG_FILE}"

      PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python dyllm/eval/eval.py \
        --batch-size "${BATCH_SIZE}" \
        --model-path "${MODEL_PATH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --num-shot "${NUM_SHOT}" \
        --tasks "${TASKS}" \
        --limit "${LIMIT}" \
        --num-steps "${NUM_STEPS}" \
        --num-full-steps "${NUM_FULL_STEPS}" \
        --threshold "${THRESHOLD}" \
        --ignore-eos \
        --block-size "${BLOCK_SIZE}" \
        2>&1 | tee -a "${LOG_FILE}"

    done
  done
done