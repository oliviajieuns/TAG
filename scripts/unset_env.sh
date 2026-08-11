#!/usr/bin/env bash
# Unset every env var we touch during debugging so the next shell starts
# from a clean slate. Source it (don't execute) so the unsets affect the
# current shell:
#
#     source scripts/unset_env.sh
#
# Mirrors setup_env.sh's variable set, plus the DDP debugging knobs,
# plus the torchrun / SLURM rendezvous variables that can leak between
# runs.

# --- torchrun / SLURM rendezvous (leak between runs causes silent exits) ---
unset RANK LOCAL_RANK WORLD_SIZE
unset MASTER_ADDR MASTER_PORT
unset GROUP_RANK LOCAL_WORLD_SIZE ROLE_RANK ROLE_NAME ROLE_WORLD_SIZE
unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
unset TORCHELASTIC_USE_AGENT_STORE TORCHELASTIC_ERROR_FILE

# --- DDP / NCCL debug knobs ---
unset TADS_DDP_BACKEND
unset TADS_DDP_FIND_UNUSED
unset TADS_DDP_BROADCAST_BUFFERS
unset TADS_DDP_STATIC_GRAPH
unset TADS_NCCL_REINIT
unset TADS_DL_NUM_WORKERS
unset TADS_ENABLE_NO_SYNC
unset TADS_ENABLE_COREDUMPS

unset TORCH_DISTRIBUTED_DEBUG
unset TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
unset TORCH_NCCL_ASYNC_ERROR_HANDLING
unset NCCL_ASYNC_ERROR_HANDLING
unset NCCL_BLOCKING_WAIT
unset NCCL_DEBUG
unset NCCL_DEBUG_SUBSYS
unset NCCL_SOCKET_IFNAME

# --- TADS data / model paths (re-source setup_env.sh to restore defaults) ---
unset MODEL_PATH_LLAMA2_7B MODEL_PATH_QWEN25_7B
unset MODEL_PATH_MISTRAL_7B MODEL_PATH_DEEPSEEK_7B
unset ALPACA_DATA_FILES ALPACA_DATASET_NAME
unset OUTPUT_ROOT DATA_CACHE EVAL_RESULTS_ROOT
unset MMLU_DATA_DIR GSM8K_DATA_DIR HUMANEVAL_DATA_DIR TYDIQA_DATA_DIR BBH_DATA_DIR

# --- HF / tokenizers ---
unset HF_HOME HF_DATASETS_CACHE HF_HUB_CACHE TRANSFORMERS_CACHE
unset HF_DATASETS_OFFLINE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
unset HF_ENDPOINT
unset TOKENIZERS_PARALLELISM TRANSFORMERS_NO_ADVISORY_WARNINGS

# --- pytorch / cuda runtime ---
unset PYTORCH_CUDA_ALLOC_CONF

echo "[unset_env] cleared TADS / NCCL / torchrun / HF env vars."
echo "[unset_env] re-source setup_env.sh to restore defaults:"
echo "             source scripts/setup_env.sh"
