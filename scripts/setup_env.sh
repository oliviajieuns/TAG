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
# The model loader does case-insensitive sibling lookup if these paths don't
# match the on-disk casing exactly (Linux is case-sensitive; HF / cluster
# naming conventions vary). So both `qwen2.5-7b` and `Qwen2.5-7B` resolve.
export MODEL_PATH_LLAMA2_7B="${MODEL_PATH_LLAMA2_7B:-/group-volume/nait-models/Llama-2-7b-hf}"
export MODEL_PATH_QWEN25_7B="${MODEL_PATH_QWEN25_7B:-/group-volume/nait-models/qwen2.5-7b}"
export MODEL_PATH_QWEN25_05B="${MODEL_PATH_QWEN25_05B:-/group-volume/nait-models/qwen2.5-0.5b}"
export MODEL_PATH_QWEN25_14B="${MODEL_PATH_QWEN25_14B:-/group-volume/jieuns/models/Qwen2.5-14B}"
export MODEL_PATH_MISTRAL_7B="${MODEL_PATH_MISTRAL_7B:-/group-volume/nait-models/mistral-7b-v0.1}"
export MODEL_PATH_DEEPSEEK_7B="${MODEL_PATH_DEEPSEEK_7B:-/group-volume/nait-models/DeepSeek-LLM-7B-Base}"

# --- IT training data (Alpaca-GPT4 local file) ---
# File extension picks the loader automatically: .parquet / .json / .jsonl / .csv.
# Default is a glob over the canonical cluster layout (the HF Alpaca-GPT4
# distribution shards into hashed filenames like
# `train-00000-of-00001-XXXX.json`, so a single-file path is brittle across
# re-downloads). Override with a concrete file path before sourcing if you
# want exact-match behaviour.
export ALPACA_DATA_FILES="${ALPACA_DATA_FILES:-/group-volume/IT-datasets/alpaca_gpt4/data/*.json}"

# --- Output roots ---
export OUTPUT_ROOT="${OUTPUT_ROOT:-/group-volume/minsoo3.kim/tads-checkpoints}"
export DATA_CACHE="${DATA_CACHE:-/group-volume/minsoo3.kim/tads-checkpoints/cache}"
export EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-/group-volume/minsoo3.kim/tads-eval-results}"

# --- Benchmark data dirs ---
export MMLU_DATA_DIR="${MMLU_DATA_DIR:-/group-volume/IT-datasets/mmlu/all}"
export GSM8K_DATA_DIR="${GSM8K_DATA_DIR:-/group-volume/IT-datasets/gsm8k}"
export HUMANEVAL_DATA_DIR="${HUMANEVAL_DATA_DIR:-/group-volume/IT-datasets/human-eval}"
export TYDIQA_DATA_DIR="${TYDIQA_DATA_DIR:-/group-volume/IT-datasets/tydiqa}"
export BBH_DATA_DIR="${BBH_DATA_DIR:-/group-volume/IT-datasets/bbh}"

# --- Runtime hygiene ---
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Core dump policy ---
# A single 7B-DDP rank that segfaults can drop a ~240 GB core file into
# the current working directory (the dump contains the process's full
# virtual address space — model weights + bnb 8-bit optimiser state +
# gradient buffers + CUDA-mapped VRAM regions). On a 4-GPU launch that
# is ~960 GB total, easily destroying the 50 GB user-volume. Disable
# coredumps by default; users who want them for debugging should opt in
# by exporting TADS_ENABLE_COREDUMPS=1 BEFORE sourcing this file and
# also `cd $OUTPUT_ROOT/coredumps` (or any group-volume dir) before
# launching, so the dump lands on the shared volume rather than user-volume.
if [ "${TADS_ENABLE_COREDUMPS:-0}" = "1" ]; then
    mkdir -p "${OUTPUT_ROOT}/coredumps" 2>/dev/null || true
    ulimit -c unlimited 2>/dev/null || true
    echo "[setup_env] coredumps ENABLED — make sure cwd is on group-volume" \
         "(e.g. cd ${OUTPUT_ROOT}/coredumps) before launching training."
else
    ulimit -c 0 2>/dev/null || true
fi
# Silence HF tokenizer's per-call "Token indices sequence length is longer
# than the specified maximum sequence length for this model" advisory.
# Our code intentionally truncates to max_seq_len; the warning is noise.
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"

# --- Offline by default ---
# Every model, tokenizer, and dataset must already be on local disk. The HF
# libs reach over the network even for local files (metadata refresh,
# dataset-card lookup, version pings); on cluster nodes without outbound
# HTTPS that turns into "tries to download → cache-lock corruption" errors
# minutes after start. To re-enable hub access for a one-off run, override
# these to "0" BEFORE running training/eval. The Python entry points set
# the same defaults via os.environ.setdefault, so an unset shell still
# behaves offline.
#
# Lifecycle note: once exported, these stick for the rest of the shell
# session. If you `source scripts/setup_env.sh` and later want a single
# online command without re-sourcing, prefix the command:
#     HF_DATASETS_OFFLINE=0 HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
#         python -m tads.eval ...
# This is local to that subprocess and doesn't disturb the parent shell.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# Optional HF mirror (only matters if you've opted back into online mode):
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
_tads_warn ALPACA_DATA_FILES      "$ALPACA_DATA_FILES"      "Alpaca-GPT4 training file (parquet / json / jsonl / csv)"

# Required for evaluation (only matter if you actually run that benchmark)
_tads_warn MMLU_DATA_DIR          "$MMLU_DATA_DIR"          "MMLU 'all' parquet directory"
_tads_warn GSM8K_DATA_DIR         "$GSM8K_DATA_DIR"         "GSM8K root (contains main/test*.parquet)"
_tads_warn HUMANEVAL_DATA_DIR     "$HUMANEVAL_DATA_DIR"     "HumanEval directory or HumanEval.jsonl.gz"
_tads_warn TYDIQA_DATA_DIR        "$TYDIQA_DATA_DIR"        "TyDiQA directory (contains tydiqa-goldp-v1.1-dev.json)"
_tads_warn BBH_DATA_DIR           "$BBH_DATA_DIR"           "BBH directory (contains <task>.json + optional cot-prompts/)"

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
