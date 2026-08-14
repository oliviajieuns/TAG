#!/usr/bin/env bash
# Self-contained environment for running TAG on a generic GPU box
# (Lambda / RunPod / vast.ai / a bare A100 VM).
#
#   source scripts/gpu_cloud/env.sh
#
# Unlike scripts/setup_env.sh — which points at the n9 cluster's
# /group-volume mounts — everything here lives under ONE workspace
# directory, so a fresh machine needs no pre-existing filesystem layout.
#
# Override the workspace before sourcing if you have a big scratch disk:
#   TAG_WORKSPACE=/mnt/data/tag source scripts/gpu_cloud/env.sh

# Resolve the repo root even when sourced from another directory.
_TAG_ENV_SRC="${BASH_SOURCE[0]:-$0}"
export TAG_REPO_ROOT="$(cd "$(dirname "$_TAG_ENV_SRC")/../.." && pwd)"

# Everything the run needs lives here: weights, datasets, pools, outputs.
# On a cluster with a per-user group volume, prefer that over the repo dir —
# $HOME is usually small and the repo may sit on an overlay fs.
_tag_pick_workspace() {
  if [ -n "${TAG_WORKSPACE:-}" ]; then echo "$TAG_WORKSPACE"; return; fi
  local u="${USER:-$(id -un 2>/dev/null)}"
  for cand in "/group-volume/$u/tag-workspace" "/mnt/data/tag-workspace"; do
    local parent
    parent="$(dirname "$cand")"
    if [ -d "$parent" ] && [ -w "$parent" ]; then echo "$cand"; return; fi
  done
  echo "$TAG_REPO_ROOT/workspace"
}
# Whether the caller pinned it, BEFORE _tag_pick_workspace fills it in.
if [ -n "${TAG_WORKSPACE:-}" ]; then _TAG_WS_PINNED=1; else _TAG_WS_PINNED=""; fi
export TAG_WORKSPACE="$(_tag_pick_workspace)"

# A workspace sitting beside the repo (../workspace) is a layout the earlier
# n9 bootstrap produced, and _tag_pick_workspace does NOT find it — it would
# silently choose /group-volume/$USER/tag-workspace instead and report every
# pool as missing, inviting a rebuild of assets that already exist. Adopt it
# when it is clearly the one in use, but never over an explicit choice.
if [ -z "$_TAG_WS_PINNED" ] && [ ! -d "$TAG_WORKSPACE/pools" ]; then
  for _cand in "$TAG_REPO_ROOT/../workspace" "$TAG_REPO_ROOT/workspace"; do
    if [ -d "$_cand/pools" ]; then
      TAG_WORKSPACE="$(cd "$_cand" && pwd)"
      export TAG_WORKSPACE
      echo "[tag-env] adopting the existing workspace beside the repo:" >&2
      echo "[tag-env]   $TAG_WORKSPACE" >&2
      echo "[tag-env] (export TAG_WORKSPACE=... before sourcing to override)" >&2
      break
    fi
  done
  unset _cand
fi

# n9_discover.sh probes the cluster for models, corpora and every benchmark
# directory and writes the answers to discovered_env.sh. Only n9_env.sh read
# it, so a shell that sourced THIS file — the documented entry point — threw
# that work away and reported benchmarks as missing that are on the box. Read
# it here instead; n9_env.sh sourcing it first is harmless because every
# assignment below is ${VAR:-default}.
for _tag_disc in "$TAG_REPO_ROOT/discovered_env.sh" "$TAG_REPO_ROOT/../discovered_env.sh"; do
  if [ -f "$_tag_disc" ]; then
    # shellcheck disable=SC1090
    . "$_tag_disc"
    [ -z "${TAG_ENV_QUIET:-}" ] && echo "[tag-env] discovered paths: $_tag_disc"
    break
  fi
done
unset _tag_disc

# Reuse assets the machine already has rather than re-downloading ~1 GB.
# First existing candidate wins; the workspace path is the download target.
_tag_first_existing() {
  # usage: _tag_first_existing <marker-file-or-empty> <candidate>...
  local marker="$1"; shift
  for c in "$@"; do
    if [ -n "$marker" ]; then
      [ -f "$c/$marker" ] && { echo "$c"; return; }
    else
      [ -f "$c" ] && { echo "$c"; return; }
    fi
  done
  echo ""
}

_TAG_M05B="$(_tag_first_existing config.json \
  "${MODEL_PATH_QWEN25_05B:-/nonexistent}" \
  "$TAG_WORKSPACE/models/qwen2.5-0.5b" \
  /group-volume/nait-models/qwen2.5-0.5b \
  /group-volume/models/Qwen2.5-0.5B \
  /group-volume/models/Qwen2.5-0.5B-Instruct)"
export MODEL_PATH_QWEN25_05B="${_TAG_M05B:-$TAG_WORKSPACE/models/qwen2.5-0.5b}"

_TAG_M7B="$(_tag_first_existing config.json \
  "${MODEL_PATH_QWEN25_7B:-/nonexistent}" \
  "$TAG_WORKSPACE/models/qwen2.5-7b" \
  /group-volume/nait-models/qwen2.5-7b \
  /group-volume/models/Qwen2.5-7B \
  /group-volume/models/Qwen2.5-7B-Instruct)"
export MODEL_PATH_QWEN25_7B="${_TAG_M7B:-$TAG_WORKSPACE/models/qwen2.5-7b}"

export POOLS="${POOLS:-$TAG_WORKSPACE/pools}"

_TAG_RAW="$(_tag_first_existing "" \
  "${ALPACA_RAW_JSON:-/nonexistent}" \
  "$TAG_WORKSPACE/datasets/alpaca_gpt4.json" \
  "/group-volume/${USER:-nobody}/datasets/alpaca_gpt4.json" \
  /group-volume/IT-datasets/alpaca_gpt4/data/train.json)"
export ALPACA_RAW_JSON="${_TAG_RAW:-$TAG_WORKSPACE/datasets/alpaca_gpt4.json}"

# The candidate pool the training run actually selects from. bootstrap.sh
# generates it; until then it does not exist and preflight will say so.
export ALPACA_DATA_FILES="${ALPACA_DATA_FILES:-$POOLS/composite20/pool.json}"
export TAG_CF_FILES="${TAG_CF_FILES:-$POOLS/composite20/counterfactual.json}"
export TAG_DEDUP_FILE="${TAG_DEDUP_FILE:-$POOLS/composite20/dedup_clusters.json}"
# The gate reference is BACKBONE-SPECIFIC: Delta_hat is a property of a
# particular model's likelihoods, so a 0.5B reference mis-scales a 7B gate.
# One shared variable made that mistake silent and easy, so each backbone
# gets its own and the arm configs read the matching one.
export TAG_GATE_REF="${TAG_GATE_REF:-$POOLS/clean_ref/delta_hat_05b.pt}"
export TAG_GATE_REF_7B="${TAG_GATE_REF_7B:-$POOLS/clean_ref/delta_hat_7b.pt}"
# The Eq. 5' ablation arm (tag_nonull_7b) needs a reference fit WITHOUT the
# null correction, and its own gate cache, because G differs.
export TAG_GATE_REF_7B_NONULL="${TAG_GATE_REF_7B_NONULL:-$POOLS/clean_ref/delta_hat_7b_nonull.pt}"
export TAG_GATE_CACHE_NONULL="${TAG_GATE_CACHE_NONULL:-$POOLS/composite20/tag_gate_qwen2.5-7b_nonull.pt}"
# The Delta_bar-only arm (tag_bar_7b, tail_mode: none) — s is a quantile of
# Delta_hat and tail_mode changes its distribution, so it needs its own
# reference and its own gate.
export TAG_GATE_REF_7B_BAR="${TAG_GATE_REF_7B_BAR:-$POOLS/clean_ref/delta_hat_7b_bar.pt}"
export TAG_GATE_CACHE_BAR="${TAG_GATE_CACHE_BAR:-$POOLS/composite20/tag_gate_qwen2.5-7b_bar.pt}"
# The prefix arm (tag_prefix_7b) — the measured-best support statistic.
export TAG_GATE_REF_7B_PREFIX="${TAG_GATE_REF_7B_PREFIX:-$POOLS/clean_ref/delta_hat_7b_prefix.pt}"
export TAG_GATE_CACHE_PREFIX="${TAG_GATE_CACHE_PREFIX:-$POOLS/composite20/tag_gate_qwen2.5-7b_prefix.pt}"

# The CLEAN-pool control pair (configs/experiments/clean/). Same corpus the
# calibration reference is fit on — bootstrap already emits pool.json and
# counterfactual.json there, so nothing new has to be generated — but its own
# gate cache, because G is a function of the pool and this is a different one
# from composite20. No dedup file: the legacy arm has none, so threading one
# into the TAG arm would make the pair differ by more than G.
export TAG_CLEAN_POOL="${TAG_CLEAN_POOL:-$POOLS/clean_ref/pool.json}"
export TAG_CLEAN_CF="${TAG_CLEAN_CF:-$POOLS/clean_ref/counterfactual.json}"
export TAG_GATE_CACHE_CLEAN="${TAG_GATE_CACHE_CLEAN:-$POOLS/clean_ref/tag_gate_qwen2.5-7b_prefix.pt}"

# Never let the HF hub be consulted for the TRAINING data — a silent hub
# fallback is how you end up training on a different pool than you think.
# bootstrap.sh flips these off only for the download step.
export ALPACA_DATASET_NAME=""
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-$TAG_WORKSPACE/hf_home}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-$TAG_WORKSPACE/runs}"
export DATA_CACHE="${DATA_CACHE:-$TAG_WORKSPACE/cache}"
export EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-$TAG_WORKSPACE/eval-results}"

# Benchmark corpora for `python -m tag.eval`. These lived only in
# scripts/setup_env.sh (the n9-specific file), so a shell that sourced THIS
# file — the documented entry point — had none of them and every eval failed
# on a missing data dir.
#
# The corpora are NOT all under one root: setup_env.sh's defaults say
# /group-volume/IT-datasets, but this box keeps them at /group-volume/datasets,
# and MMLU sits one level deeper (mmlu/all) in some layouts and not in others.
# Probing a single root reported eight benchmarks as missing that were all
# present. So: try every root we have seen, and for each benchmark try the
# spellings it actually appears under. An explicit export always wins, and
# TAG_BENCH_ROOTS prepends more roots without editing this file.
_TAG_BENCH_ROOTS="${TAG_BENCH_ROOTS:-} $TAG_WORKSPACE/datasets /group-volume/datasets /group-volume/IT-datasets /group-volume/data/datasets /group-volume/${USER:-nobody}/datasets"

_tag_bench_dir() {  # usage: _tag_bench_dir VARNAME subdir [subdir...]
  local var="$1"; shift
  local cur="${!var:-}"
  # An explicit export wins — but only if it exists, so a stale value copied
  # between machines does not shadow a corpus that IS here.
  [ -n "$cur" ] && [ -d "$cur" ] && { echo "$cur"; return; }
  # Spelling is the OUTER loop: the more specific one (mmlu/all) must beat a
  # bare mmlu found under an earlier root, or MMLU resolves to a directory
  # whose parquet files are one level down and the evaluator finds nothing.
  local root sub
  for sub in "$@"; do
    for root in $_TAG_BENCH_ROOTS; do
      # Non-empty, not merely present: an empty directory left behind by an
      # interrupted download would otherwise shadow the real corpus.
      if [ -d "$root/$sub" ] && [ -n "$(ls -A "$root/$sub" 2>/dev/null)" ]; then
        echo "$root/$sub"; return
      fi
    done
  done
  # Nothing found: report the preferred spelling under the download target so
  # the message names a path the download scripts will actually create.
  echo "${cur:-$TAG_WORKSPACE/datasets/$1}"
}
export MMLU_DATA_DIR="$(_tag_bench_dir MMLU_DATA_DIR mmlu/all mmlu)"
export MMLU_PRO_DATA_DIR="$(_tag_bench_dir MMLU_PRO_DATA_DIR mmlu_pro mmlu-pro)"
export GSM8K_DATA_DIR="$(_tag_bench_dir GSM8K_DATA_DIR gsm8k)"
export SVAMP_DATA_DIR="$(_tag_bench_dir SVAMP_DATA_DIR svamp SVAMP)"
export HUMANEVAL_DATA_DIR="$(_tag_bench_dir HUMANEVAL_DATA_DIR human-eval humaneval human_eval)"
export MBPP_DATA_DIR="$(_tag_bench_dir MBPP_DATA_DIR mbpp)"
export TYDIQA_DATA_DIR="$(_tag_bench_dir TYDIQA_DATA_DIR tydiqa tydi_qa)"
export XQUAD_DATA_DIR="$(_tag_bench_dir XQUAD_DATA_DIR xquad)"
export BBH_DATA_DIR="$(_tag_bench_dir BBH_DATA_DIR bbh BIG-Bench-Hard bbh/bbh)"

# Forward-only batch size for the pool scoring passes. The config default
# (_shared_7b.yaml: ${oc.env:TAG_EPISODE_BS_7B,8}) is sized for a small GPU,
# and until now the 32 was set only in n9_env.sh — so a shell that sourced
# THIS file directly silently ran the gate precompute at bs=8 and took four
# times as long. Twice. The default belongs where every entry point sees it.
export TAG_EPISODE_BS="${TAG_EPISODE_BS:-64}"        # 0.5B
export TAG_EPISODE_BS_7B="${TAG_EPISODE_BS_7B:-32}"  # 7B

# The PCA in TrajectoryAnchor.update runs torch.linalg.eigh per layer; on
# hosts with many cores each call spawns OMP_NUM_THREADS workers in tight
# succession and can blow past the container's pids limit.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$TAG_WORKSPACE"/{models,datasets,pools,runs,cache,hf_home}

_tag_mark() { [ -e "$1" ] && echo "found" || echo "MISSING (bootstrap will create)"; }

if [ -z "${TAG_ENV_QUIET:-}" ]; then
  echo "[tag-env] workspace : $TAG_WORKSPACE"
  echo "[tag-env] model 0.5b: $MODEL_PATH_QWEN25_05B  [$(_tag_mark "$MODEL_PATH_QWEN25_05B/config.json")]"
  echo "[tag-env] model 7b  : $MODEL_PATH_QWEN25_7B  [$(_tag_mark "$MODEL_PATH_QWEN25_7B/config.json")]"
  case "$MODEL_PATH_QWEN25_7B" in
    *Instruct*|*instruct*)
      echo "[tag-env] NOTE: the 7B checkpoint is an INSTRUCT model, not a base"
      echo "[tag-env]       model. That is a real deviation from 'SFT from a base"
      echo "[tag-env]       checkpoint' and must be stated in any writeup that"
      echo "[tag-env]       mixes it with base-model numbers."
      ;;
  esac
  echo "[tag-env] raw corpus: $ALPACA_RAW_JSON  [$(_tag_mark "$ALPACA_RAW_JSON")]"
  echo "[tag-env] pool      : $ALPACA_DATA_FILES  [$(_tag_mark "$ALPACA_DATA_FILES")]"
  echo "[tag-env] gate ref 0.5b: $TAG_GATE_REF  [$(_tag_mark "$TAG_GATE_REF")]"
  echo "[tag-env] gate ref 7b  : $TAG_GATE_REF_7B  [$(_tag_mark "$TAG_GATE_REF_7B")]"
  echo "[tag-env] outputs   : $OUTPUT_ROOT"
  echo "[tag-env] eval out  : $EVAL_RESULTS_ROOT"
  echo "[tag-env] fwd batch : 0.5b=$TAG_EPISODE_BS  7b=$TAG_EPISODE_BS_7B"
  # The paper's Table 2 is these eight, in this order.
  _tag_have=""; _tag_missing=""
  for _pair in mmlu:MMLU_DATA_DIR bbh:BBH_DATA_DIR svamp:SVAMP_DATA_DIR \
               gsm8k:GSM8K_DATA_DIR mbpp:MBPP_DATA_DIR \
               humaneval:HUMANEVAL_DATA_DIR tydiqa:TYDIQA_DATA_DIR \
               xquad:XQUAD_DATA_DIR; do
    _b="${_pair%%:*}"; _v="${_pair##*:}"
    if [ -d "${!_v}" ]; then _tag_have="$_tag_have $_b"
    else _tag_missing="$_tag_missing $_b"; fi
  done
  echo "[tag-env] benchmarks:${_tag_have:- none} found"
  if [ -n "$_tag_missing" ]; then
    echo "[tag-env]   MISSING:$_tag_missing — eval will refuse to start on these"
    echo "[tag-env]   roots searched:$_TAG_BENCH_ROOTS"
    echo "[tag-env]   if a corpus is on this box under another path, either"
    echo "[tag-env]   export TAG_BENCH_ROOTS=/its/parent before sourcing, or"
    echo "[tag-env]   run: bash scripts/gpu_cloud/n9_discover.sh --write"
    echo "[tag-env]   Otherwise fetch it with scripts/download_<bench>.sh."
  fi
  # The resolution above picks between several spellings per benchmark, and
  # picking the wrong one fails deep inside eval rather than here.
  if [ -n "${TAG_ENV_VERBOSE:-}" ]; then
    for _pair in mmlu:MMLU_DATA_DIR bbh:BBH_DATA_DIR svamp:SVAMP_DATA_DIR \
                 gsm8k:GSM8K_DATA_DIR mbpp:MBPP_DATA_DIR \
                 humaneval:HUMANEVAL_DATA_DIR tydiqa:TYDIQA_DATA_DIR \
                 xquad:XQUAD_DATA_DIR; do
      _b="${_pair%%:*}"; _v="${_pair##*:}"
      printf '[tag-env]   %-10s %s\n' "$_b" "${!_v}"
    done
  else
    echo "[tag-env]   (TAG_ENV_VERBOSE=1 prints the resolved path of each)"
  fi
  unset _pair _b _v _tag_have _tag_missing
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[tag-env] gpus      : $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
  fi
  echo "[tag-env] next      : bash scripts/gpu_cloud/bootstrap.sh && python scripts/gpu_cloud/preflight.py"

  # If the shell's cwd belongs to a DIFFERENT git repo than the one this
  # env.sh came from, every relative command the runbook gives ("git pull",
  # "bash scripts/...") lands somewhere else. That has already happened once:
  # the TAG clone sat one level in, and `git pull origin main` run from its
  # parent updated an unrelated repo while reporting success.
  if command -v git >/dev/null 2>&1; then
    _tag_here="$(git -C . rev-parse --show-toplevel 2>/dev/null || true)"
    _tag_repo="$(git -C "$TAG_REPO_ROOT" rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$_tag_repo" ] && [ "$_tag_here" != "$_tag_repo" ]; then
      echo "" >&2
      echo "[tag-env] WARNING: your shell is NOT inside the TAG checkout." >&2
      echo "[tag-env]   cwd git root : ${_tag_here:-<not a git repo>}" >&2
      echo "[tag-env]   TAG git root : $_tag_repo" >&2
      echo "[tag-env] 'git pull' and 'bash scripts/...' from here will hit the" >&2
      echo "[tag-env] wrong tree. Run:  cd $_tag_repo" >&2
    fi
    unset _tag_here _tag_repo
  fi
fi
