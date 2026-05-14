#!/usr/bin/env bash
# Source this file once per shell before running training / evaluation:
#   source scripts/setup_env.sh
#
# - Exports every path the YAML configs reference via ${oc.env:...}.
# - Creates output directories.
# - **Warns** (does not error) about any input path that doesn't exist on
#   this filesystem, and prints exact override commands for each one.
#
# This script never aborts your shell. If you only need a subset of the
# models / benchmarks, just ignore the warnings for the rest.

# -----------------------------------------------------------------------------
# Defaults (override any of these BEFORE sourcing to skip a warning)
# -----------------------------------------------------------------------------

# --- LLM checkpoints ---
export MODEL_PATH_LLAMA2_7B="${MODEL_PATH_LLAMA2_7B:-/group-volume/nait-models/Llama-2-7b-hf}"
export MODEL_PATH_QWEN25_7B="${MODEL_PATH_QWEN25_7B:-/group-volume/nait-models/qwen2.5-7b}"
export MODEL_PATH_MISTRAL_7B="${MODEL_PATH_MISTRAL_7B:-/group-volume/nait-models/Mistral-7B-v0.1}"
export MODEL_PATH_DEEPSEEK_7B="${MODEL_PATH_DEEPSEEK_7B:-/group-volume/nait-models/deepseek-llm-7b-base}"

# --- IT training data (Alpaca-GPT4 local parquet) ---
export ALPACA_DATA_FILES="${ALPACA_DATA_FILES:-/group-volume/IT-datasets/alpaca_gpt4/train.parquet}"

# --- Output roots ---
export OUTPUT_ROOT="${OUTPUT_ROOT:-/group-volume/minsoo3.kim/tads-checkpoints}"
export DATA_CACHE="${DATA_CACHE:-/group-volume/minsoo3.kim/tads-checkpoints/cache}"
export EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-/group-volume/minsoo3.kim/tads-eval-results}"

# --- Benchmark data dirs ---
export MMLU_DATA_DIR="${MMLU_DATA_DIR:-/group-volume/IT-datasets/mmlu/all}"
export GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-/group-volume/IT-datasets/gsm8k}"
export HUMANEVAL_DATA_DIR="${HUMANEVAL_DATA_DIR:-/group-volume/IT-datasets/human-eval}"
export TYDIQA_DATA_DIR="${TYDIQA_DATA_DIR:-/group-volume/IT-datasets/tydiqa}"

# --- Runtime hygiene ---
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Optional HF mirror (uncomment if you need hub access):
# export HF_ENDPOINT=https://hf-mirror.com

# -----------------------------------------------------------------------------
# Create output dirs (idempotent, never warns)
# -----------------------------------------------------------------------------
mkdir -p "$OUTPUT_ROOT" "$DATA_CACHE" "$EVAL_RESULTS_ROOT" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Existence checks (warn-only — never aborts the shell)
# -----------------------------------------------------------------------------
_tads_missing=0
_tads_warn() {
    local var="$1" path="$2" desc="$3"
    if [ ! -e "$path" ]; then
        if [ "$_tads_missing" = "0" ]; then
            echo ""
            echo "------------------------------------------------------------------"
            echo "[setup_env] WARNINGS: the following paths do not exist locally."
            echo "[setup_env] Override the env var BEFORE sourcing this file,"
            echo "[setup_env] or set it manually after sourcing."
            echo "------------------------------------------------------------------"
        fi
        printf "  [missing] %-25s %s\n" "$var" "$path"
        printf "            (%s)\n" "$desc"
        printf "            fix:  export %s=/your/path\n" "$var"
        _tads_missing=$((_tads_missing + 1))
    fi
}

# Required for training (any one of the four models you actually plan to use)
_tads_warn MODEL_PATH_LLAMA2_7B   "$MODEL_PATH_LLAMA2_7B"   "Llama-2-7B base checkpoint dir"
_tads_warn MODEL_PATH_QWEN25_7B   "$MODEL_PATH_QWEN25_7B"   "Qwen2.5-7B base checkpoint dir"
_tads_warn MODEL_PATH_MISTRAL_7B  "$MODEL_PATH_MISTRAL_7B"  "Mistral-7B-v0.1 base checkpoint dir"
_tads_warn MODEL_PATH_DEEPSEEK_7B "$MODEL_PATH_DEEPSEEK_7B" "DeepSeek-LLM-7B base checkpoint dir"
_tads_warn ALPACA_DATA_FILES      "$ALPACA_DATA_FILES"      "Alpaca-GPT4 training parquet"

# Required for evaluation (only matter if you actually run that benchmark)
_tads_warn MMLU_DATA_DIR          "$MMLU_DATA_DIR"          "MMLU 'all' parquet directory"
_tads_warn GSM8K_DATA_DIR         "$GSM8K_DATA_DIR"         "GSM8K root (contains main/test*.parquet)"
_tads_warn HUMANEVAL_DATA_DIR     "$HUMANEVAL_DATA_DIR"     "HumanEval directory or HumanEval.jsonl.gz"
_tads_warn TYDIQA_DATA_DIR        "$TYDIQA_DATA_DIR"        "TyDiQA directory (contains tydiqa-goldp-v1.1-dev.json)"

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------
if [ "$_tads_missing" -gt 0 ]; then
    echo "------------------------------------------------------------------"
    echo "[setup_env] $_tads_missing path(s) missing."
    echo "[setup_env] Env vars are still exported (with their default values)"
    echo "[setup_env] — fix the missing ones before running training/eval."
    echo "------------------------------------------------------------------"
else
    echo "[setup_env] All paths verified ✓"
fi
echo ""
echo "TADS env loaded."
echo "  OUTPUT_ROOT       = $OUTPUT_ROOT"
echo "  EVAL_RESULTS_ROOT = $EVAL_RESULTS_ROOT"
echo "  ALPACA_DATA_FILES = $ALPACA_DATA_FILES"

unset -f _tads_warn
unset _tads_missing
