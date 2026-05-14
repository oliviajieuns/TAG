#!/usr/bin/env bash
# Source this file once per shell before running training / evaluation:
#   source scripts/setup_env.sh
#
# Centralises every path that the YAML configs reference via ${oc.env:...}.

# --- LLM checkpoints (local, downloaded ahead of time) ---
export MODEL_PATH_LLAMA2_7B=/group-volume/nait-models/Llama-2-7b-hf
export MODEL_PATH_QWEN25_7B=/group-volume/nait-models/qwen2.5-7b
export MODEL_PATH_MISTRAL_7B=/group-volume/nait-models/Mistral-7B-v0.1
export MODEL_PATH_DEEPSEEK_7B=/group-volume/nait-models/deepseek-llm-7b-base

# --- IT training data (Alpaca-GPT4 local parquet) ---
export ALPACA_DATA_FILES=/group-volume/IT-datasets/alpaca_gpt4/train.parquet

# --- Output roots ---
export OUTPUT_ROOT=/group-volume/minsoo3.kim/tads-checkpoints
export DATA_CACHE=/group-volume/minsoo3.kim/tads-checkpoints/cache
export EVAL_RESULTS_ROOT=/group-volume/minsoo3.kim/tads-eval-results

# --- Benchmark data dirs ---
export MMLU_DATA_DIR=/group-volume/IT-datasets/mmlu/all
export GSM8K_DATA_DIR=/group-volume/IT-datasets/gsm8k
export HUMANEVAL_DATA_DIR=/group-volume/IT-datasets/human-eval
export TYDIQA_DATA_DIR=/group-volume/IT-datasets/tydiqa

# --- Runtime hygiene ---
export TOKENIZERS_PARALLELISM=false
# Reduce CUDA memory fragmentation under long-running DDP jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional Hugging Face mirror (uncomment if needed for hub access):
# export HF_ENDPOINT=https://hf-mirror.com

mkdir -p "$OUTPUT_ROOT" "$DATA_CACHE" "$EVAL_RESULTS_ROOT" 2>/dev/null || true

echo "TADS env loaded."
echo "  OUTPUT_ROOT       = $OUTPUT_ROOT"
echo "  EVAL_RESULTS_ROOT = $EVAL_RESULTS_ROOT"
echo "  ALPACA_DATA_FILES = $ALPACA_DATA_FILES"
