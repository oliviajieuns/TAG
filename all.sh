#!/usr/bin/env bash

# Finish the tuned TAG eight-benchmark evaluation.  The quick evaluator
# (eval.sh) owns MBPP/GSM8K/MMLU; this queue spreads the remaining five over
# every currently idle visible GPU and merges the two validated result sets.
#
# raeval.sh reuses this same hardened queue engine for a fresh R x A repeat;
# that mode evaluates all eight benchmarks from scratch.
#
#   S=42 bash all.sh
#   S=42 bash all.sh status

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run with: bash all.sh" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=$(cd "$(dirname "$0")" && pwd)
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
FRESH=${TAG_WORKSPACE:-/group-volume/jieuns.shin/tag2/workspace}
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
WRAPPER="$REPO/scripts/gpu_cloud/tag_eval_he_spawn_serial.py"
WRAPPER_SHA=0ace9b83c9179f0ff8f07f79576586c1f383ee058a8cd42d4f0cc383b2b8bc4a

die() {
  echo "STOP: $*" >&2
  exit 2
}

resolve() {
  SEED=${S:-42}
  ARM=${A:-tag}
  case "$SEED" in
    1|7|42) ;;
    *) die "S must be 1, 7, or 42" ;;
  esac

  case "$ARM" in
    tag)
      CFG="$REPO/configs/experiments/main_7b/llama2/tag_10_schedfloor_bs64.yaml"
      CELL="$FRESH/runs/main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}/runs"
      RUN_GLOB=tag10tune_
      TASK_COUNT=5
      TASK_PREFIX=rest5
      EXPERIMENT=llama2_tag_10_schedfloor_bs64
      ;;
    ra)
      CFG="$REPO/configs/experiments/main_7b/llama2/legacy_10.yaml"
      CELL="$FRESH/runs/main_7b/llama2/legacy_10_repeat_seed${SEED}/runs"
      RUN_GLOB=ra_
      TASK_COUNT=8
      TASK_PREFIX=all8part
      EXPERIMENT=llama2_legacy_10
      ;;
    *) die "A must be tag or ra" ;;
  esac

  RUN=""
  local candidate
  shopt -s nullglob
  for candidate in "$CELL"/${RUN_GLOB}*; do
    [[ -f "$candidate/TRAIN_VALIDATED" ]] || continue
    if [[ -z "$RUN" || "$candidate/TRAIN_VALIDATED" -nt "$RUN/TRAIN_VALIDATED" ]]; then
      RUN=$candidate
    fi
  done
  [[ -n "$RUN" ]] || die "no validated $ARM training run for seed $SEED"

  CKPT="$RUN/epoch_last"
  RUN_NAME=$(basename "$RUN")
  QUEUE="$FRESH/eval-queue/${ARM}_eval_seed${SEED}_${RUN_NAME}"
  SHARDS="$FRESH/eval-shards/main_7b/llama2/${ARM}/seed${SEED}/${RUN_NAME}"
  LOG_ROOT="$FRESH/logs/table2_${ARM}_eval/seed${SEED}_${RUN_NAME}"
  QUICK_DIR="$FRESH/eval-results/main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}_quick3/runs/quick3_${RUN_NAME}"
  FINAL_TAG="all8_${RUN_NAME}"
  if [[ "$ARM" == tag ]]; then
    FINAL_BASE="$FRESH/eval-results/main_7b/llama2/tag_10_schedfloor_bs64_seed${SEED}"
  else
    FINAL_BASE="$FRESH/eval-results/main_7b/llama2/legacy_10_repeat_seed${SEED}"
  fi
  FINAL_DIR="$FINAL_BASE/runs/$FINAL_TAG"
}

check_repo() {
  cd "$REPO"
  git diff --quiet || die "tracked code changes present"
  git diff --cached --quiet || die "staged code changes present"
}

check_training() {
  [[ "$(<"$CKPT/_complete")" == 3 ]] || die "training checkpoint is not epoch 3"
  [[ ! -e "$CKPT/_save_errors.json" ]] || die "training checkpoint has save errors"
  "$PY" - "$RUN" "$SEED" "$ARM" <<'PY'
import json, math, sys
from pathlib import Path

run, seed, arm = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed and cfg["training_mode"] == "full"
assert cfg["method"] == "selection" and cfg["selection_ratio"] == 0.1
assert cfg.get("adamw_foreach") is False
selection = cfg["selection"]
if arm == "tag":
    assert cfg["grad_accum"] == 4
    assert cfg["batch_size"] * cfg["launch_world_size"] * cfg["grad_accum"] == 64
    assert math.isclose(float(cfg["min_lr_ratio"]), 0.10)
    assert selection["score_mode"] == "tag"
else:
    assert cfg["grad_accum"] == 8
    assert cfg["batch_size"] * cfg["launch_world_size"] * cfg["grad_accum"] == 128
    assert math.isclose(float(cfg.get("min_lr_ratio", 0.0)), 0.0)
    assert math.isclose(float(cfg["warmup_ratio"]), 0.06)
    assert math.isclose(float(cfg["gradient_clip"]), 0.5)
    assert selection.get("score_mode", "legacy") == "legacy"
    assert math.isclose(float(selection["lam"]), 1.0) and selection["use_anchor"] is True
print("TRAIN_INPUT_OK", run / "epoch_last")
PY
}

check_static_inputs() {
  [[ -x "$PY" ]] || die "python missing"
  [[ -f "$CFG" ]] || die "config missing"
  [[ -f "$WRAPPER" ]] || die "HumanEval wrapper missing"
  [[ "$(sha256sum "$WRAPPER" | awk '{print $1}')" == "$WRAPPER_SHA" ]] || \
    die "HumanEval wrapper SHA mismatch"
  local paths=(
    "$OLD/eval-data/mmlu"
    "$OLD/eval-data/bbh"
    "$OLD/eval-data/humaneval/HumanEval.jsonl.gz"
    "$OLD/eval-data/tydiqa"
    "$OLD/eval-data/xquad"
    /group-volume/datasets/svamp/data
    /group-volume/datasets/gsm8k/datasets/openai/gsm8k
    /group-volume/datasets/mbpp
  )
  local path
  for path in "${paths[@]}"; do
    [[ -e "$path" ]] || die "eval input missing: $path"
  done

  PYTHONPATH="$REPO" "$PY" - "$OLD" <<'PY'
import gzip, json, sys
from pathlib import Path
from tag.evals.bbh import _list_task_files, _load_cot_prefix
from tag.evals.tydiqa import _resolve_split_paths

old = Path(sys.argv[1])
bbh = old / "eval-data/bbh"
tasks = _list_task_files(bbh)
assert len(tasks) == 27
assert all(_load_cot_prefix(bbh, task.stem) is not None for task in tasks)
with gzip.open(old / "eval-data/humaneval/HumanEval.jsonl.gz", "rt") as f:
    he = [json.loads(line) for line in f if line.strip()]
assert len(he) == 164
assert {"task_id", "prompt", "entry_point", "test"} <= set(he[0])
dev, train, _ = _resolve_split_paths(str(old / "eval-data/tydiqa"))
assert Path(dev).is_file() and Path(train).is_file()
xquad = old / "eval-data/xquad"
langs = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]
assert all((xquad / f"xquad.{lang}.json").is_file() for lang in langs)
print("EVAL8_CORPORA_OK mmlu bbh=27+cot svamp gsm8k mbpp humaneval=164 tydiqa=dev+train xquad=12")
PY
}

prepare_queue() {
  resolve
  check_repo
  check_training
  check_static_inputs
  mkdir -p "$(dirname "$QUEUE")"
  exec 8>"${QUEUE}.prepare.lock"
  flock 8

  if [[ -f "$QUEUE/PREPARED" ]]; then
    local n
    n=$(find "$QUEUE/tasks" -maxdepth 1 -type f -name '*.task' | wc -l)
    [[ "$n" -eq "$TASK_COUNT" ]] || die "queue has $n tasks, expected $TASK_COUNT"
    flock -u 8
    exec 8>&-
    return 0
  fi
  [[ ! -e "$QUEUE" ]] || die "partial queue exists: $QUEUE"

  local tmp="${QUEUE}.build.$(hostname -s).$$"
  mkdir -p "$tmp/tasks" "$tmp/claims" "$tmp/done" "$tmp/failed" "$tmp/nodes"
  local rows
  if [[ "$ARM" == tag ]]; then
    rows=(
      '001 humaneval' '002 xquad' '003 bbh' '004 tydiqa' '005 svamp'
    )
  else
    rows=(
      '001 humaneval' '002 xquad' '003 bbh' '004 mmlu'
      '005 gsm8k' '006 tydiqa' '007 mbpp' '008 svamp'
    )
  fi
  local row priority bench name
  for row in "${rows[@]}"; do
    read -r priority bench <<<"$row"
    name="${priority}_${bench}"
    printf '%s\n' "$bench" >"$tmp/tasks/$name.task"
  done
  printf 'arm=%s\nseed=%s\nrun=%s\ntasks=%s\nprepared_utc=%s\n' \
    "$ARM" "$SEED" "$RUN_NAME" "$TASK_COUNT" "$(date -u +%FT%TZ)" >"$tmp/PREPARED"
  mv "$tmp" "$QUEUE"
  echo "QUEUE_PREPARED arm=$ARM seed=$SEED tasks=$TASK_COUNT"
  flock -u 8
  exec 8>&-
}

idle_gpus() {
  local index free
  while IFS=',' read -r index free; do
    index=${index//[[:space:]]/}
    free=${free//[[:space:]]/}
    if [[ "$free" =~ ^[0-9]+$ && "$free" -ge 78000 ]]; then
      printf '%s\n' "$index"
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
  )
}

task_tag() {
  printf '%s_%s_%s\n' "$TASK_PREFIX" "$RUN_NAME" "$1"
}

task_run_dir() {
  local bench=$1 tag
  tag=$(task_tag "$bench")
  printf '%s\n' "$SHARDS/$bench/runs/$tag"
}

validate_task() {
  local run=$1 bench=$2 tag=$3 log=$4
  "$PY" - "$run" "$bench" "$tag" "$CKPT" "$SEED" "$log" <<'PY'
import collections, gzip, json, math, sys
from pathlib import Path

run, bench, tag, ckpt = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
seed, log = int(sys.argv[5]), Path(sys.argv[6])
assert (run / "_complete").read_text().strip() == tag
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed and cfg["ckpt"] == ckpt
train_cfg = json.loads((Path(ckpt).parent / "cfg.json").read_text())
assert cfg.get("git_sha") == train_cfg.get("git_sha")
summaries = list(run.glob("*-eval_summary.json"))
results = list(run.glob(f"*-{bench}.json"))
assert len(summaries) == len(results) == 1, (summaries, results)
payload, result = json.loads(summaries[0].read_text()), json.loads(results[0].read_text())
assert payload["limit"] is None and payload["failures"] == []
assert len(payload["summaries"]) == 1 and payload["summaries"][0]["benchmark"] == bench
assert result["benchmark"] == bench
score = 100.0 * float(result["accuracy"])
assert math.isfinite(score) and 0.0 <= score <= 100.0
assert math.isclose(float(payload["summaries"][0]["accuracy"]), float(result["accuracy"]), rel_tol=0, abs_tol=1e-12)

if bench == "mmlu":
    assert (result["num_subjects"], result["total_questions"]) == (57, 14042)
elif bench == "bbh":
    assert (result["num_tasks"], result["tasks_with_official_cot_prompt"]) == (27, 27)
    assert result["total_questions"] == 6511 and result["generation_batch_size"] == 16
elif bench == "svamp":
    # The mounted ChilleD mirror currently exposes a 300-item test parquet;
    # other canonical layouts expose all 1,000. Record and accept either,
    # but never silently validate a debug subset.
    assert result["total"] in {300, 1000}, result["total"]
    assert result["generation_batch_size"] == 16
elif bench == "gsm8k":
    assert result["total"] == 1319 and result["generation_batch_size"] == 16
elif bench == "mbpp":
    assert result["total_questions"] == 257 and result["generation_batch_size"] == 16
    assert result["config"] == "sanitized" and result["n_samples"] == 1
    assert result["n_fewshot"] == 3
elif bench == "tydiqa":
    assert result["total"] == 5077 and result["n_fewshot"] == 5
    assert result["paper_faithful"] is True and result["n_silent_zero_shot"] == 0
    assert len(result["per_language"]) == 9 and result["generation_batch_size"] == 16
elif bench == "xquad":
    assert result["total_questions"] == 14275 and result["n_fewshot"] == 5
    assert len(result["languages"]) == 12 and result["missing_languages"] == []
    assert result["generation_batch_size"] == 16
elif bench == "humaneval":
    assert (result["num_problems"], result["n_samples"], result["n_total_samples"]) == (164, 20, 3280)
    assert result["primary_metric"] == "pass@10"
    assert result["max_new_tokens"] == 512 and result["temperature"] == 0.8
    assert result["top_p"] == 0.95 and result["accuracy"] == result["pass@10"]
    assert 0 <= result["n_truncated_samples"] <= 3280
    assert len(result["first_samples"]) == 3
    comp = Path(result["completions_file"])
    scored = Path(str(comp) + "_results.jsonl")
    assert comp.is_file() and scored.is_file()
    def rows(path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    completions, scores = rows(comp), rows(scored)
    assert len(completions) == len(scores) == 3280
    cc = collections.Counter(x["task_id"] for x in completions)
    rc = collections.Counter(x["task_id"] for x in scores)
    assert len(cc) == 164 and set(cc.values()) == {20} and rc == cc
    assert all(type(x.get("passed")) is bool and "result" in x for x in scores)
    correct = collections.Counter(x["task_id"] for x in scores if x["passed"])
    def estimate(n, c, k):
        return 1.0 if n - c < k else 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))
    p10 = sum(estimate(20, correct[t], 10) for t in cc) / 164
    p1 = sum(estimate(20, correct[t], 1) for t in cc) / 164
    assert math.isclose(result["pass@10"], p10, rel_tol=0, abs_tol=1e-10)
    assert math.isclose(result["pass@1"], p1, rel_tol=0, abs_tol=1e-10)
    text = log.read_text(errors="replace")
    assert "[tag-safe-he]" in text and "start_method=spawn" in text
    assert "n_workers=1" in text and "filelock_loaded=False" in text
    failures = collections.Counter(
        str(x.get("result", "missing")) for x in scores if not x["passed"]
    )
    top_failures = "; ".join(
        f"{reason}={count}" for reason, count in failures.most_common(5)
    )
    diag = (
        f"HUMANEVAL_DIAG pass@1={100*p1:.2f} pass@10={100*p10:.2f} "
        f"truncated={result['n_truncated_samples']}/3280 "
        f"failed_samples={sum(failures.values())}/3280"
    )
    (run / "HUMANEVAL_DIAG.txt").write_text(
        diag + "\nHUMANEVAL_TOP_FAILURES " + top_failures + "\n"
    )
    print(diag)
print(f"EVAL_TASK_VALIDATED seed={seed} bench={bench} score={score:.2f}")
PY
}

run_task() {
  local task=$1 gpu=$2 bench tag base run log rc entry
  read -r bench <"$QUEUE/tasks/$task.task"
  tag=$(task_tag "$bench")
  base="$SHARDS/$bench"
  run="$base/runs/$tag"
  log="$LOG_ROOT/tasks/${bench}.log"
  mkdir -p "$base" "$(dirname "$log")"

  echo "TASK_START task=$task host=$(hostname -s) gpu=$gpu bench=$bench utc=$(date -u +%FT%TZ)"
  if [[ -e "$run" ]]; then
    if [[ ! -f "$run/_complete" ]]; then
      echo "incomplete task run exists: $run" >&2
      return 2
    fi
    if ! validate_task "$run" "$bench" "$tag" "$log"; then
      echo "TASK_REUSE_VALIDATION_FAILED task=$task run=$run" >&2
      return 3
    fi
    echo "TASK_REUSED task=$task"
    return 0
  fi

  entry=("$PY" -m tag.eval)
  if [[ "$bench" == humaneval ]]; then
    entry=("$PY" "$WRAPPER")
  fi

  rc=0
  env -u LD_PRELOAD -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO" \
    HF_HOME="$FRESH/hf_home" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    TAG_EVAL_GEN_BS=16 \
    TAG_GSM8K_USE_SFT_WRAP=0 \
    TAG_MBPP_USE_SFT_WRAP=0 \
    "${entry[@]}" \
      --config "$CFG" \
      --ckpt "$CKPT" \
      --benchmarks "$bench" \
      --out_dir "$base" \
      --eval_tag "$tag" \
      --training_mode full \
      --cuda_device 0 \
      --mmlu_data_dir "$OLD/eval-data/mmlu" \
      --bbh_data_dir "$OLD/eval-data/bbh" \
      --svamp_data_dir /group-volume/datasets/svamp/data \
      --gsm8k_data_dir /group-volume/datasets/gsm8k/datasets/openai/gsm8k \
      --mbpp_data_dir /group-volume/datasets/mbpp \
      --humaneval_data_dir "$OLD/eval-data/humaneval" \
      --tydiqa_data_dir "$OLD/eval-data/tydiqa" \
      --xquad_data_dir "$OLD/eval-data/xquad" \
      >"$log" 2>&1 || rc=$?
  if [[ "$rc" -ne 0 ]]; then
    echo "TASK_EVAL_FAILED task=$task rc=$rc log=$log" >&2
    return "$rc"
  fi
  if ! validate_task "$run" "$bench" "$tag" "$log"; then
    echo "TASK_RESULT_VALIDATION_FAILED task=$task run=$run log=$log" >&2
    return 3
  fi
  echo "TASK_COMPLETE task=$task host=$(hostname -s) gpu=$gpu utc=$(date -u +%FT%TZ)"
}

claim_task() {
  local file name
  shopt -s nullglob
  for file in "$QUEUE/tasks"/*.task; do
    name=$(basename "$file" .task)
    [[ ! -e "$QUEUE/done/$name" && ! -e "$QUEUE/failed/$name" ]] || continue
    if mkdir "$QUEUE/claims/$name" 2>/dev/null; then
      printf '%s\n' "$name"
      return 0
    fi
  done
  return 1
}

recover_stale_claims_for_host() {
  local host=$1 claim name owner_host bench run stamp log
  mkdir -p "$QUEUE/stale_claims" "$LOG_ROOT/stale"
  shopt -s nullglob
  for claim in "$QUEUE/claims"/*; do
    [[ -d "$claim" ]] || continue
    name=$(basename "$claim")
    [[ ! -e "$QUEUE/done/$name" && ! -e "$QUEUE/failed/$name" ]] || continue
    [[ -f "$claim/owner" ]] || continue
    owner_host=$(awk -F= '$1 == "host" {print $2}' "$claim/owner")
    [[ "$owner_host" == "$host" ]] || continue
    read -r bench <"$QUEUE/tasks/$name.task"
    run=$(task_run_dir "$bench")
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ -e "$run" ]]; then
      mv "$run" "${run}.stale.${host}.${stamp}"
    fi
    log="$LOG_ROOT/tasks/${bench}.log"
    if [[ -f "$log" ]]; then
      mv "$log" "$LOG_ROOT/stale/${bench}.${host}.${stamp}.log"
    fi
    mv "$claim" "$QUEUE/stale_claims/${name}.${host}.${stamp}"
    echo "STALE_TASK_REQUEUED task=$name prior_host=$host"
  done
}

lane_worker() {
  local gpu=$1 task rc lane_rc=0
  while task=$(claim_task); do
    printf 'host=%s\ngpu=%s\npid=%s\nclaimed_utc=%s\n' \
      "$(hostname -s)" "$gpu" "$$" "$(date -u +%FT%TZ)" >"$QUEUE/claims/$task/owner"
    rc=0
    run_task "$task" "$gpu" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
      printf 'host=%s\ngpu=%s\ncompleted_utc=%s\n' \
        "$(hostname -s)" "$gpu" "$(date -u +%FT%TZ)" >"$QUEUE/done/$task"
    else
      printf 'host=%s\ngpu=%s\nrc=%s\nfailed_utc=%s\n' \
        "$(hostname -s)" "$gpu" "$rc" "$(date -u +%FT%TZ)" >"$QUEUE/failed/$task"
      lane_rc=1
    fi
  done
  echo "LANE_DRAINED host=$(hostname -s) gpu=$gpu"
  return "$lane_rc"
}

finalize_results() {
  local done_n failed_n rc=0
  done_n=$(find "$QUEUE/done" -mindepth 1 -maxdepth 1 -type f | wc -l)
  failed_n=$(find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f | wc -l)
  [[ "$failed_n" -eq 0 ]] || return 1
  [[ "$done_n" -eq "$TASK_COUNT" ]] || return 0

  exec 7>"$QUEUE/finalize.lock"
  flock -n 7 || return 0

  "$PY" - "$QUEUE" "$SHARDS" "$QUICK_DIR" "$FINAL_DIR" "$FINAL_TAG" \
    "$CKPT" "$SEED" "$CFG" "$RUN_NAME" "$ARM" "$TASK_PREFIX" \
    "$EXPERIMENT" <<'PY' || rc=$?
import json, math, os, shutil, statistics, sys
from datetime import datetime
from pathlib import Path

queue, shards, quick, final = map(Path, sys.argv[1:5])
final_tag, ckpt, seed, cfg, run_name = sys.argv[5], sys.argv[6], int(sys.argv[7]), sys.argv[8], sys.argv[9]
arm, task_prefix, experiment = sys.argv[10:13]
rest = ["humaneval", "xquad", "bbh", "tydiqa", "svamp"]
canonical = ["mmlu", "bbh", "svamp", "gsm8k", "mbpp", "humaneval", "tydiqa", "xquad"]

def result_path(root, bench):
    found = list(root.glob(f"*-{bench}.json"))
    assert len(found) == 1, (root, bench, found)
    return found[0]

task_benches = rest if arm == "tag" else canonical
task_scores = {}
for bench in task_benches:
    tag = f"{task_prefix}_{run_name}_{bench}"
    run = shards / bench / "runs" / tag
    assert (run / "_complete").read_text().strip() == tag
    task_scores[bench] = 100.0 * float(json.loads(result_path(run, bench).read_text())["accuracy"])
task_text = " ".join(f"{b.upper()}={task_scores[b]:.2f}" for b in task_benches)
(queue / "TASK_SCORES.txt").write_text(task_text + "\n")
print("EVAL_TASKS_VALIDATED", task_text)

if arm == "tag" and not (quick / "EVAL_VALIDATED").is_file():
    print("WAITING_FOR_QUICK3", quick)
    raise SystemExit(3)

if final.exists():
    assert (final / "_complete").read_text().strip() == final_tag
    existing = (final / "ALL8_SCORES.txt").read_text()
    (queue / "ALL8_SCORES.txt").write_text(existing)
    print(existing.strip())
    raise SystemExit(0)

sources = {}
if arm == "tag":
    for bench in ("mmlu", "gsm8k", "mbpp"):
        sources[bench] = quick
    for bench in rest:
        sources[bench] = shards / bench / "runs" / f"{task_prefix}_{run_name}_{bench}"
else:
    for bench in canonical:
        sources[bench] = shards / bench / "runs" / f"{task_prefix}_{run_name}_{bench}"

final.parent.mkdir(parents=True, exist_ok=True)
stage = final.parent / f".{final.name}.build.{os.getpid()}"
assert not stage.exists()
stage.mkdir()
summaries, source_runs = [], {}
for bench in canonical:
    src = sources[bench]
    path = result_path(src, bench)
    result = json.loads(path.read_text())
    assert result["benchmark"] == bench and math.isfinite(float(result["accuracy"]))
    if bench == "humaneval":
        comp = Path(result["completions_file"])
        scored = Path(str(comp) + "_results.jsonl")
        assert comp.is_file() and scored.is_file()
        dst_comp = stage / comp.name
        dst_scores = stage / scored.name
        shutil.copy2(comp, dst_comp)
        shutil.copy2(scored, dst_scores)
        diag = src / "HUMANEVAL_DIAG.txt"
        assert diag.is_file()
        shutil.copy2(diag, stage / diag.name)
        result["completions_file"] = str(final / dst_comp.name)
    (stage / path.name).write_text(json.dumps(result, indent=2) + "\n")
    summaries.append(result)
    source_runs[bench] = str(src)

train_cfg = json.loads((Path(ckpt).parent / "cfg.json").read_text())
(stage / "cfg.json").write_text(json.dumps({
    "seed": seed,
    "git_sha": train_cfg["git_sha"],
    "ckpt": ckpt,
    "config": cfg,
    "eval_arm": arm,
}, indent=2) + "\n")
(stage / f"{experiment}-eval_summary.json").write_text(json.dumps({
    "experiment": experiment,
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "ckpt": ckpt,
    "base_model": train_cfg["model_path"],
    "limit": None,
    "prompt_style": train_cfg.get("prompt_style", "alpaca_default"),
    "summaries": summaries,
    "failures": [],
    "source_runs": source_runs,
}, indent=2) + "\n")

scores = {item["benchmark"]: 100.0 * float(item["accuracy"]) for item in summaries}
macro = statistics.fmean(scores.values())
score_text = " ".join(f"{b.upper()}={scores[b]:.2f}" for b in canonical)
score_text += f" AVG={macro:.2f}"
(stage / "ALL8_SCORES.txt").write_text(score_text + "\n")
(stage / "_complete").write_text(final_tag + "\n")
os.rename(stage, final)
(queue / "ALL8_SCORES.txt").write_text(score_text + "\n")
print("ALL8_EVAL_VALIDATED", score_text)
PY
  if [[ "$rc" -eq 3 ]]; then
    return 0
  fi
  [[ "$rc" -eq 0 ]] || return "$rc"
  printf 'finalized_utc=%s\nfinal_dir=%s\n' \
    "$(date -u +%FT%TZ)" "$FINAL_DIR" >"$QUEUE/FINALIZED"
}

node_worker() {
  resolve
  [[ -f "$QUEUE/PREPARED" ]] || die "queue not prepared"
  [[ -n "${TAG_REST_GPUS:-}" ]] || die "no GPU assignment"
  IFS=',' read -r -a gpus <<<"$TAG_REST_GPUS"
  local host gpu free rc=0 pid
  host=$(hostname -s)
  for gpu in "${gpus[@]}"; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    [[ "$free" -ge 78000 ]] || die "GPU $gpu became busy"
  done

  mkdir -p "$QUEUE/nodes" "$LOG_ROOT/nodes" "$LOG_ROOT/lanes"
  exec 9>"$QUEUE/nodes/$host.lock"
  flock -n 9 || die "node worker already active on $host"
  recover_stale_claims_for_host "$host"
  printf 'pid=%s\nstarted_utc=%s\ngpus=%s\n' \
    "$$" "$(date -u +%FT%TZ)" "${gpus[*]}" >"$QUEUE/nodes/$host.running"

  on_node_exit() {
    local exit_rc=$?
    rm -f "$QUEUE/nodes/$host.running"
    printf 'finished_utc=%s\nrc=%s\n' \
      "$(date -u +%FT%TZ)" "$exit_rc" >"$QUEUE/nodes/$host.finished"
    echo "NODE_END host=$host rc=$exit_rc utc=$(date -u +%FT%TZ)"
  }
  trap on_node_exit EXIT

  export TAG_WORKSPACE="$FRESH"
  export HF_HOME="$FRESH/hf_home"
  export TAG_ENV_RESET=1
  source "$REPO/scripts/gpu_cloud/env.sh"
  unset TAG_ENV_RESET
  export PYTHONHASHSEED=0

  cd "$REPO"
  env -u LD_PRELOAD -u RANK -u LOCAL_RANK -u WORLD_SIZE \
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$REPO" \
    "$PY" "$WRAPPER" --smoke-human-eval-scorer

  echo "NODE_START host=$host arm=$ARM seed=$SEED gpus=${gpus[*]} utc=$(date -u +%FT%TZ)"
  local pids=()
  for gpu in "${gpus[@]}"; do
    lane_worker "$gpu" >"$LOG_ROOT/lanes/${host}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  finalize_results || rc=1
  return "$rc"
}

launch() {
  prepare_queue
  mapfile -t gpus < <(idle_gpus)
  [[ "${#gpus[@]}" -gt 0 ]] || die "no idle GPU (need at least 78000 MiB free)"
  local csv host log pid
  csv=$(IFS=,; echo "${gpus[*]}")
  host=$(hostname -s)
  log="$LOG_ROOT/nodes/${host}.log"
  mkdir -p "$(dirname "$log")"
  TAG_REST_GPUS="$csv" nohup bash "$0" --worker >"$log" 2>&1 </dev/null &
  pid=$!
  sleep 4
  if kill -0 "$pid" 2>/dev/null; then
    echo "STARTED arm=$ARM seed=$SEED host=$host gpus=$csv"
    if [[ "$ARM" == tag ]]; then
      echo "ORDER=HumanEval,XQuAD,BBH,TyDiQA,SVAMP"
      echo "STATUS: S=$SEED bash all.sh status"
    else
      echo "ORDER=HumanEval,XQuAD,BBH,MMLU,GSM8K,TyDiQA,MBPP,SVAMP"
      echo "STATUS: S=$SEED bash raeval.sh status"
    fi
  else
    echo "START_FAILED"
    tail -n 100 "$log" || true
    exit 1
  fi
}

status() {
  resolve
  [[ -d "$QUEUE" ]] || die "queue not started"
  finalize_results >/dev/null 2>&1 || true
  local total claimed done_n failed_n
  total=$(find "$QUEUE/tasks" -maxdepth 1 -type f -name '*.task' | wc -l)
  claimed=$(find "$QUEUE/claims" -mindepth 1 -maxdepth 1 -type d | wc -l)
  done_n=$(find "$QUEUE/done" -mindepth 1 -maxdepth 1 -type f | wc -l)
  failed_n=$(find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f | wc -l)
  echo "STATUS total=$total claimed=$claimed validated=$done_n failed=$failed_n"
  echo "IN_FLIGHT=$((claimed - done_n - failed_n))"
  if [[ "$failed_n" -gt 0 ]]; then
    echo "FAILED"
    find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort
  fi
  [[ ! -f "$QUEUE/TASK_SCORES.txt" ]] || cat "$QUEUE/TASK_SCORES.txt"
  if [[ "$ARM" == tag && "$done_n" -eq "$TASK_COUNT" && ! -f "$QUICK_DIR/EVAL_VALIDATED" ]]; then
    echo "WAITING_FOR_QUICK3=$QUICK_DIR"
  fi
  local he_run
  he_run=$(task_run_dir humaneval)
  [[ ! -f "$he_run/HUMANEVAL_DIAG.txt" ]] || cat "$he_run/HUMANEVAL_DIAG.txt"
  if [[ -f "$QUEUE/ALL8_SCORES.txt" ]]; then
    echo "STATUS=ALL8_EVAL_VALIDATED"
    cat "$QUEUE/ALL8_SCORES.txt"
  fi
  echo "RECENT"
  local file
  for file in "$LOG_ROOT"/tasks/*.log; do
    [[ -e "$file" ]] || continue
    printf '%s  ' "$(basename "$file")"
    tail -n 1 "$file"
  done
}

case "${1:-}" in
  --worker) node_worker ;;
  status) status ;;
  "") launch ;;
  *) die "usage: S={1|7|42} bash all.sh [status]" ;;
esac
