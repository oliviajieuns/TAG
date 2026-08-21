#!/usr/bin/env bash

# Launch the 24 Table-2 TAG evaluation cells (3 seeds x 8 benchmarks) on
# the shared 2+4+4 GPU allocation.  Run --prepare once, then run this script
# once on each of the three allocated nodes.  The default mode detaches.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "STOP: run this file with: bash ${BASH_SOURCE[0]}" >&2
  return 2
fi

set -Eeuo pipefail
umask 002

REPO=/group-volume/jieuns.shin/tag2
FRESH="$REPO/workspace"
OLD=/group-volume/jieuns.shin/tads/tests/tag/workspace
PY=/group-volume/jieuns.shin/venvs/exp/bin/python
PIN=5cef83473550f6a20fb349f088c0444d2da7abaf
CFG="$REPO/configs/experiments/main_7b/llama2/tag_10.yaml"
WRAPPER="$FRESH/tools/tag_eval_he_spawn_serial.py"
WRAPPER_SHA=0ace9b83c9179f0ff8f07f79576586c1f383ee058a8cd42d4f0cc383b2b8bc4a
QUEUE="$FRESH/eval-queue/tag10_5cef834_eval8_v1"
SHARDS="$FRESH/eval-shards/main_7b/llama2/tag_10"
LOG_ROOT="$FRESH/logs/table2_tag_eval"
FINAL_ROOT="$FRESH/eval-results"
EVAL_TAG_PREFIX=tag10_5cef834
BENCHES=(mmlu bbh svamp gsm8k mbpp humaneval tydiqa xquad)

die() {
  echo "STOP: $*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

check_repo() {
  cd "$REPO"
  [[ "$(git rev-parse HEAD)" == "$PIN" ]] || die "repo HEAD is not $PIN"
  git diff --quiet || die "tracked worktree changes present under $REPO"
  git diff --cached --quiet || die "staged changes present under $REPO"
}

checkpoint_for_seed() {
  case "$1" in
    1)  printf '%s\n' "$FRESH/runs/main_7b/llama2/tag_10_seed1/runs/tag10_5cef834_seed1_2g_b8_ga8_foreach0_ncclfix/epoch_last" ;;
    7)  printf '%s\n' "$FRESH/runs/main_7b/llama2/tag_10_seed7/runs/tag10_5cef834_seed7_4g_b4_ga8_foreach0_ncclfix/epoch_last" ;;
    42) printf '%s\n' "$FRESH/runs/main_7b/llama2/tag_10_seed42/runs/tag10_5cef834_seed42_4g_b4_ga8_foreach0_ncclfix/epoch_last" ;;
    *) return 2 ;;
  esac
}

task_run_dir() {
  local seed=$1 bench=$2
  local base="$SHARDS/seed${seed}/${bench}"
  local tag="${EVAL_TAG_PREFIX}_seed${seed}_${bench}_full"
  printf '%s\n' "$base/runs/$tag"
}

check_static_inputs() {
  [[ -x "$PY" ]] || die "python missing: $PY"
  [[ -f "$CFG" ]] || die "config missing: $CFG"
  [[ -f "$WRAPPER" ]] || die "HumanEval wrapper missing: $WRAPPER"
  [[ "$(sha256_file "$WRAPPER")" == "$WRAPPER_SHA" ]] || die "HumanEval wrapper SHA mismatch"
  [[ -f /group-volume/models/Llama-2-7b-hf/config.json ]] || die "Llama-2 model missing"
  local paths=(
    "$OLD/eval-data/mmlu"
    "$OLD/eval-data/bbh"
    /group-volume/datasets/svamp/data
    /group-volume/datasets/gsm8k/datasets/openai/gsm8k
    /group-volume/datasets/mbpp
    "$OLD/eval-data/humaneval"
    "$OLD/eval-data/tydiqa"
    "$OLD/eval-data/xquad"
  )
  local path
  for path in "${paths[@]}"; do
    [[ -e "$path" ]] || die "eval input missing: $path"
  done
  PYTHONPATH="$REPO" "$PY" - "$OLD" <<'PY'
import sys
from pathlib import Path
from tag.evals.bbh import _list_task_files, _load_cot_prefix
from tag.evals.tydiqa import _resolve_split_paths

old = Path(sys.argv[1])
bbh = old / "eval-data/bbh"
tasks = _list_task_files(bbh)
assert len(tasks) == 27, (len(tasks), bbh)
assert all(_load_cot_prefix(bbh, task.stem) is not None for task in tasks)
dev, train, _ = _resolve_split_paths(str(old / "eval-data/tydiqa"))
assert Path(dev).is_file() and Path(train).is_file(), (dev, train)
xquad = old / "eval-data/xquad"
langs = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]
assert all((xquad / f"xquad.{lang}.json").is_file() for lang in langs)
he = old / "eval-data/humaneval/HumanEval.jsonl.gz"
assert he.is_file(), he
print("EVAL_CORPUS_PREFLIGHT_OK bbh=27+cot tydiqa=dev+train xquad=12 humaneval=164-file")
PY
}

audit_training() {
  "$PY" - "$FRESH" "$OLD" "$PIN" <<'PY'
import hashlib, json, math, sys
from pathlib import Path

w, old, pin = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = {1: (2, 8, 8), 7: (4, 4, 8), 42: (4, 4, 8)}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

old_gate = sha(old / "pools/alpaca_gpt4/tag_gate_llama2-7b_prefix32.pt")
selection_hashes = []
for seed, (world, batch, grad_accum) in spec.items():
    tag = f"tag10_5cef834_seed{seed}_{world}g_b{batch}_ga{grad_accum}_foreach0_ncclfix"
    run = w / f"runs/main_7b/llama2/tag_10_seed{seed}/runs/{tag}"
    ckpt = run / "epoch_last"
    assert (run / "TRAIN_VALIDATED").is_file(), run
    assert (ckpt / "_complete").read_text().strip() == "3", ckpt
    assert not (ckpt / "_save_errors.json").exists(), ckpt
    cfg = json.loads((run / "cfg.json").read_text())
    assert cfg["seed"] == seed and cfg["git_sha"] == pin
    assert cfg["launch_world_size"] == world and cfg["batch_size"] == batch
    assert cfg["grad_accum"] == grad_accum and world * batch * grad_accum == 128\n    assert cfg.get("adamw_foreach") is False and cfg["use_8bit_optimizer"] is False
    assert cfg["method"] == "selection" and cfg["selection_ratio"] == 0.1
    assert cfg["train_epochs"] == 3 and cfg["training_mode"] == "full"
    assert cfg["model_path"] == "/group-volume/models/Llama-2-7b-hf"
    assert cfg["data_files"] == str(old / "pools/alpaca_gpt4/pool.json")
    tag_cfg = cfg["selection"]["tag"]
    assert cfg["selection"]["score_mode"] == "tag"
    assert tag_cfg["counterfactual_data_files"] == str(old / "pools/alpaca_gpt4/counterfactual.json")
    assert tag_cfg["gate_ref_file"] == str(old / "pools/clean_ref/delta_hat_llama2_prefix.pt")
    assert (tag_cfg["prefix_tokens"], tag_cfg["span_tokens"]) == (32, 16)
    assert (tag_cfg["tau"], tag_cfg["tau_mode"], tag_cfg["tail_mode"]) == (0.5, "per_token", "none")
    assert tag_cfg["null_correction"] is True and tag_cfg["target_zero_rate"] == 0.05
    assert sha(Path(tag_cfg["gate_cache_file"])) == old_gate
    metrics = json.loads((run / "metrics.json").read_text())
    assert [row["epoch"] for row in metrics] == [1, 2, 3]
    for row in metrics:
        assert row["n_total"] == 52002 and row["selected_n"] == 5200
        assert row["score_mode"] == "tag" and row["n_zero_weight_selected"] == 0
        assert 0.045 <= row["gate_zero_frac"] <= 0.055
        assert math.isfinite(row["train_loss"])
    for epoch in (1, 2, 3):
        p = run / f"selected_indices_epoch{epoch}.json"
        ids = json.loads(p.read_text())
        assert len(ids) == len(set(ids)) == 5200
        assert all(type(x) is int and 0 <= x < 52002 for x in ids)
        selection_hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    weights = list(ckpt.glob("*.safetensors"))
    assert weights and sum(p.stat().st_size for p in weights) > 10_000_000_000
    print("TRAIN_AUDITED", seed, ckpt)
assert len(set(selection_hashes)) > 1, "all selections byte-identical: suspicious"
print("ALL_3_TRAININGS_AUDITED")
PY
}

prepare_queue() {
  check_repo
  check_static_inputs
  audit_training

  if [[ -f "$QUEUE/PREPARED" ]]; then
    local n
    n=$(find "$QUEUE/tasks" -maxdepth 1 -type f -name '*.task' | wc -l)
    [[ "$n" -eq 24 ]] || die "existing queue has $n tasks, expected 24"
    echo "QUEUE_ALREADY_PREPARED=$QUEUE"
    return 0
  fi
  [[ ! -e "$QUEUE" ]] || die "partial queue already exists: $QUEUE"

  local tmp="${QUEUE}.build.$(hostname -s).$$"
  mkdir -p "$tmp/tasks" "$tmp/claims" "$tmp/done" "$tmp/failed" "$tmp/nodes"
  local rows=(
    '001 1 humaneval' '002 7 humaneval' '003 42 humaneval'
    '004 1 bbh'       '005 7 bbh'       '006 42 bbh'
    '007 1 xquad'     '008 7 xquad'     '009 42 xquad'
    '010 1 mmlu'      '011 7 mmlu'      '012 42 mmlu'
    '013 1 gsm8k'     '014 7 gsm8k'     '015 42 gsm8k'
    '016 1 tydiqa'    '017 7 tydiqa'    '018 42 tydiqa'
    '019 1 mbpp'      '020 7 mbpp'      '021 42 mbpp'
    '022 1 svamp'     '023 7 svamp'     '024 42 svamp'
  )
  local row priority seed bench name
  for row in "${rows[@]}"; do
    read -r priority seed bench <<<"$row"
    name="${priority}_seed${seed}_${bench}"
    printf '%s %s\n' "$seed" "$bench" >"$tmp/tasks/$name.task"
  done
  printf 'pin=%s\ntasks=24\nprepared_host=%s\nprepared_utc=%s\n' \
    "$PIN" "$(hostname -s)" "$(date -u +%FT%TZ)" >"$tmp/PREPARED"
  mkdir -p "$(dirname "$QUEUE")"
  mv "$tmp" "$QUEUE"
  echo "QUEUE_PREPARED=$QUEUE"
  echo "NEXT: run this script once on each of the 2+4+4 GPU nodes."
}

validate_task() {
  local run=$1 bench=$2 seed=$3 ckpt=$4 tag=$5 log=$6
  "$PY" - "$run" "$bench" "$seed" "$ckpt" "$tag" "$PIN" "$log" <<'PY'
import collections, gzip, json, math, sys
from pathlib import Path

run, bench = Path(sys.argv[1]), sys.argv[2]
seed, ckpt, tag, pin, log = int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], Path(sys.argv[7])
assert (run / "_complete").read_text().strip() == tag
cfg = json.loads((run / "cfg.json").read_text())
assert cfg["seed"] == seed and cfg["ckpt"] == ckpt and cfg.get("git_sha") == pin
sp = run / "llama2_tag_10-eval_summary.json"
bp = run / f"llama2_tag_10-{bench}.json"
payload, result = json.loads(sp.read_text()), json.loads(bp.read_text())
assert payload["limit"] is None and payload["ckpt"] == ckpt and payload["failures"] == []
assert len(payload["summaries"]) == 1 and payload["summaries"][0]["benchmark"] == bench
assert result["benchmark"] == bench
accuracy = float(result["accuracy"])
assert math.isfinite(accuracy) and 0.0 <= accuracy <= 1.0
assert abs(accuracy - float(payload["summaries"][0]["accuracy"])) < 1e-12
if bench in {"bbh", "svamp", "gsm8k", "mbpp", "tydiqa", "xquad"}:
    assert result["generation_batch_size"] == 16
if bench == "mmlu":
    assert (result["num_subjects"], result["total_questions"]) == (57, 14042)
elif bench == "bbh":
    assert (result["num_tasks"], result["tasks_with_official_cot_prompt"], result["total_questions"]) == (27, 27, 6511)
elif bench == "svamp":
    assert result["total"] == 1000
elif bench == "gsm8k":
    assert result["total"] == 1319
elif bench == "mbpp":
    assert (result["config"], result["total_questions"], result["n_samples"], result["n_fewshot"]) == ("sanitized", 257, 1, 3)
elif bench == "humaneval":
    assert (result["num_problems"], result["n_samples"], result["n_total_samples"], result["primary_metric"]) == (164, 20, 3280, "pass@10")
    assert result["max_new_tokens"] == 512 and result["temperature"] == 0.8 and result["top_p"] == 0.95
    assert result["accuracy"] == result["pass@10"] and "pass@10" in result["raw_pass_at_k"]
    comp = Path(result["completions_file"])
    scored = Path(str(comp) + "_results.jsonl")
    assert comp.is_file() and scored.is_file()
    assert comp.parent.resolve() == run.resolve()
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
    assert "[tag-safe-he]" in text and "start_method=spawn" in text and "n_workers=1" in text
    assert "filelock_loaded=False" in text
elif bench == "tydiqa":
    assert result["total"] == 5077 and result["n_fewshot"] == 5
    assert result["paper_faithful"] is True and result["n_silent_zero_shot"] == 0
    assert len(result["per_language"]) == 9
elif bench == "xquad":
    assert len(result["languages"]) == 12 and result["missing_languages"] == []
    assert result["total_questions"] == 14275 and result["n_fewshot"] == 5
print(f"EVAL_TASK_VALIDATED seed={seed} bench={bench} score={100*accuracy:.4f}")
PY
}

run_task() {
  local task_name=$1 gpu=$2
  local task_file="$QUEUE/tasks/$task_name.task"
  local seed bench
  read -r seed bench <"$task_file"
  local ckpt base tag run log
  ckpt=$(checkpoint_for_seed "$seed")
  base="$SHARDS/seed${seed}/${bench}"
  tag="${EVAL_TAG_PREFIX}_seed${seed}_${bench}_full"
  run="$base/runs/$tag"
  log="$LOG_ROOT/tasks/seed${seed}_${bench}.log"
  mkdir -p "$base" "$(dirname "$log")"

  echo "TASK_START task=$task_name host=$(hostname -s) gpu=$gpu seed=$seed bench=$bench utc=$(date -u +%FT%TZ)"
  if [[ -e "$run" ]]; then
    echo "Existing run found; validating instead of overwriting: $run"
    validate_task "$run" "$bench" "$seed" "$ckpt" "$tag" "$log" || return $?
    return
  fi

  local entry=("$PY" -m tag.eval)
  if [[ "$bench" == humaneval ]]; then
    entry=("$PY" "$WRAPPER")
  fi

  local rc=0
  env -u LD_PRELOAD -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO" \
    MODEL_PATH_LLAMA2_7B=/group-volume/models/Llama-2-7b-hf \
    OUTPUT_ROOT="$FRESH/runs" \
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
      --ckpt "$ckpt" \
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
    echo "TASK_EVAL_FAILED task=$task_name rc=$rc log=$log" >&2
    return "$rc"
  fi
  validate_task "$run" "$bench" "$seed" "$ckpt" "$tag" "$log" || return $?
  echo "TASK_COMPLETE task=$task_name host=$(hostname -s) gpu=$gpu utc=$(date -u +%FT%TZ)"
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
  local host=$1 claim name owner_host seed bench run stamp log
  mkdir -p "$QUEUE/stale_claims" "$LOG_ROOT/stale"
  shopt -s nullglob
  for claim in "$QUEUE/claims"/*; do
    [[ -d "$claim" ]] || continue
    name=$(basename "$claim")
    [[ ! -e "$QUEUE/done/$name" && ! -e "$QUEUE/failed/$name" ]] || continue
    [[ -f "$claim/owner" ]] || continue
    owner_host=$(awk -F= '$1 == "host" {print $2}' "$claim/owner")
    [[ "$owner_host" == "$host" ]] || continue
    read -r seed bench <"$QUEUE/tasks/$name.task"
    run=$(task_run_dir "$seed" "$bench")
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    if [[ -e "$run" ]]; then
      mv "$run" "${run}.stale.${host}.${stamp}"
    fi
    log="$LOG_ROOT/tasks/seed${seed}_${bench}.log"
    if [[ -f "$log" ]]; then
      mv "$log" "$LOG_ROOT/stale/seed${seed}_${bench}.${host}.${stamp}.log"
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
  local done_n failed_n
  done_n=$(find "$QUEUE/done" -mindepth 1 -maxdepth 1 -type f | wc -l)
  failed_n=$(find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f | wc -l)
  [[ "$failed_n" -eq 0 ]] || return 1
  [[ "$done_n" -eq 24 ]] || return 0

  exec 8>"$QUEUE/finalize.lock"
  flock -n 8 || return 0
  [[ ! -f "$QUEUE/FINALIZED" ]] || return 0

  "$PY" - "$FRESH" "$PIN" "$CFG" <<'PY' || return 1
import json, math, os, shutil, sys
from datetime import datetime
from pathlib import Path

fresh, pin, cfg = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
benches = ["mmlu", "bbh", "svamp", "gsm8k", "mbpp", "humaneval", "tydiqa", "xquad"]
spec = {
    1: "tag10_5cef834_seed1_2g_b8_ga8_foreach0_ncclfix",
    7: "tag10_5cef834_seed7_4g_b4_ga8_foreach0_ncclfix",
    42: "tag10_5cef834_seed42_4g_b4_ga8_foreach0_ncclfix",
}
manifest = []
for seed, train_tag in spec.items():
    ckpt = fresh / f"runs/main_7b/llama2/tag_10_seed{seed}/runs/{train_tag}/epoch_last"
    final_tag = f"tag10_5cef834_seed{seed}_8task"
    parent = fresh / f"eval-results/main_7b/llama2/tag_10_seed{seed}/runs"
    final = parent / final_tag
    parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        assert (final / "_complete").read_text().strip() == final_tag
        cfg_existing = json.loads((final / "cfg.json").read_text())
        summary_existing = json.loads((final / "llama2_tag_10-eval_summary.json").read_text())
        assert cfg_existing["seed"] == seed and cfg_existing["git_sha"] == pin
        assert cfg_existing["ckpt"] == str(ckpt) and summary_existing["ckpt"] == str(ckpt)
        assert summary_existing["failures"] == [] and summary_existing["limit"] is None
        assert [x["benchmark"] for x in summary_existing["summaries"]] == benches
        for item in summary_existing["summaries"]:
            bp = final / f"llama2_tag_10-{item['benchmark']}.json"
            result = json.loads(bp.read_text())
            assert result["benchmark"] == item["benchmark"]
            assert math.isfinite(float(result["accuracy"]))
            assert abs(float(result["accuracy"]) - float(item["accuracy"])) < 1e-12
            if item["benchmark"] == "humaneval":
                comp = Path(result["completions_file"])
                assert comp.is_file() and Path(str(comp) + "_results.jsonl").is_file()
        manifest.append({"set": "main_7b", "model": "llama2", "method": "tag_10", "seed": seed, "run_dir": str(final)})
        print("MERGED_SEED_ALREADY_VALID", seed, final)
        continue
    stage = parent / f".{final_tag}.build.{os.getpid()}"
    assert not stage.exists()
    stage.mkdir()
    summaries = []
    source_shards = {}
    for bench in benches:
        shard_tag = f"tag10_5cef834_seed{seed}_{bench}_full"
        shard = fresh / f"eval-shards/main_7b/llama2/tag_10/seed{seed}/{bench}/runs/{shard_tag}"
        assert (shard / "_complete").read_text().strip() == shard_tag
        sp = shard / "llama2_tag_10-eval_summary.json"
        bp = shard / f"llama2_tag_10-{bench}.json"
        payload, result = json.loads(sp.read_text()), json.loads(bp.read_text())
        assert payload["limit"] is None and payload["failures"] == []
        assert payload["ckpt"] == str(ckpt) and len(payload["summaries"]) == 1
        assert result["benchmark"] == bench and math.isfinite(float(result["accuracy"]))
        if bench == "humaneval":
            src_comp = Path(result["completions_file"])
            src_scores = Path(str(src_comp) + "_results.jsonl")
            dst_comp = stage / src_comp.name
            dst_scores = stage / (src_comp.name + "_results.jsonl")
            shutil.copy2(src_comp, dst_comp)
            shutil.copy2(src_scores, dst_scores)
            result["completions_file"] = str(final / dst_comp.name)
        shutil.copy2(bp, stage / bp.name)
        if bench == "humaneval":
            (stage / bp.name).write_text(json.dumps(result, indent=2) + "\n")
        summaries.append(result)
        source_shards[bench] = str(shard)
    cfg_payload = {"seed": seed, "git_sha": pin, "ckpt": str(ckpt), "config": cfg}
    (stage / "cfg.json").write_text(json.dumps(cfg_payload, indent=2) + "\n")
    combined = {
        "experiment": "llama2_tag_10",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "base_model": "/group-volume/models/Llama-2-7b-hf",
        "limit": None,
        "prompt_style": "alpaca_default",
        "summaries": summaries,
        "failures": [],
        "source_shards": source_shards,
    }
    (stage / "llama2_tag_10-eval_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    (stage / "_complete").write_text(final_tag + "\n")
    os.rename(stage, final)
    manifest.append({"set": "main_7b", "model": "llama2", "method": "tag_10", "seed": seed, "run_dir": str(final)})
    print("MERGED_SEED", seed, final)

manifest_path = fresh / "eval-results/tag10_5cef834_3seed_8task_manifest.json"
tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
tmp.write_text(json.dumps(manifest, indent=2) + "\n")
os.replace(tmp, manifest_path)
print("MANIFEST", manifest_path)
PY

  local manifest="$FINAL_ROOT/tag10_5cef834_3seed_8task_manifest.json"
  local out="$FINAL_ROOT/tag10_5cef834_3seed_8task"
  cd "$REPO"
  "$PY" scripts/make_table_v2.py \
    --manifest "$manifest" \
    --benches mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad \
    --tsv "$out.tsv" >"$out.md" || return 1
  "$PY" - "$manifest" "$REPO/scripts" >"$out.final_row.txt" <<'PY' || return 1
import statistics, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import make_table_v2 as table

benches = "mmlu,bbh,svamp,gsm8k,mbpp,humaneval,tydiqa,xquad".split(",")
rows = table.load_manifest(Path(sys.argv[1]), benches)
table.compute_macros(rows, benches, False)
table.check_no_duplicate_cells(rows)
table.check_unique_run_dirs(rows)
assert len(rows) == 3 and sorted(row.seed for row in rows) == [1, 7, 42]
task_means = [statistics.fmean(row.accs[b] for row in rows) for b in benches]
avg, sd, ci = table.mean_sd_ci([row.macro for row in rows])
delta_rounded_ref = 100.0 * (avg - 39.68) / 39.68
print("TASK_MEANS", dict(zip(benches, task_means)))
print(f"AVG={avg:.8f} SD={sd:.8f} CI=[{ci[0]:.8f},{ci[1]:.8f}]")
print(f"DELTA_VS_DISPLAYED_FULLFT_39.68={delta_rounded_ref:+.8f}%")
print("LATEX", " & ".join([*(f"{x:.2f}" for x in task_means), f"{avg:.2f}", f"{sd:.2f}", f"[{ci[0]:.2f}, {ci[1]:.2f}]", f"{delta_rounded_ref:+.2f}\\% "]))
PY
  printf 'finalized_utc=%s\nmanifest=%s\nrow=%s\n' \
    "$(date -u +%FT%TZ)" "$manifest" "$out.final_row.txt" >"$QUEUE/FINALIZED"
  echo "ALL_24_EVALS_VALIDATED_AND_FINALIZED"
  cat "$out.final_row.txt"
}

node_worker() {
  [[ -f "$QUEUE/PREPARED" ]] || die "queue not prepared; run --prepare first"
  check_repo
  check_static_inputs
  local host expected gpu_count
  host=$(hostname -s)
  case "$host" in
    run269897*) expected=2 ;;
    run270622*) expected=4 ;;
    run270630*) expected=4 ;;
    *) die "unexpected node: $host" ;;
  esac
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  gpu_count=${#gpus[@]}
  [[ "$gpu_count" -eq "$expected" ]] || die "$host sees $gpu_count GPU(s), expected $expected"

  mkdir -p "$QUEUE/nodes" "$LOG_ROOT/nodes" "$LOG_ROOT/lanes"
  exec 9>"$QUEUE/nodes/$host.lock"
  flock -n 9 || die "a node worker is already active on $host"
  recover_stale_claims_for_host "$host"
  printf 'pid=%s\nstarted_utc=%s\ngpus=%s\n' "$$" "$(date -u +%FT%TZ)" "${gpus[*]}" >"$QUEUE/nodes/$host.running"

  cd "$REPO"
  env -u LD_PRELOAD -u RANK -u LOCAL_RANK -u WORLD_SIZE \
    CUDA_VISIBLE_DEVICES='' PYTHONPATH="$REPO" \
    "$PY" "$WRAPPER" --smoke-human-eval-scorer

  echo "NODE_WORKER_START host=$host gpus=${gpus[*]} utc=$(date -u +%FT%TZ)"
  local pids=() gpu pid rc=0
  for gpu in "${gpus[@]}"; do
    lane_worker "$gpu" >"$LOG_ROOT/lanes/${host}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  rm -f "$QUEUE/nodes/$host.running"
  finalize_results || rc=1
  printf 'finished_utc=%s\nrc=%s\n' "$(date -u +%FT%TZ)" "$rc" >"$QUEUE/nodes/$host.finished"
  echo "NODE_WORKER_END host=$host rc=$rc utc=$(date -u +%FT%TZ)"
  return "$rc"
}

launch_node() {
  [[ -f "$QUEUE/PREPARED" ]] || die "queue not prepared; run --prepare once first"
  local host log pid
  host=$(hostname -s)
  log="$LOG_ROOT/nodes/${host}.log"
  mkdir -p "$(dirname "$log")"
  nohup bash "$0" --node-worker >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  echo "STARTED host=$host pid=$pid"
  echo "LOG=$log"
  echo "Run this same file once on each remaining allocated node."
}

show_status() {
  [[ -d "$QUEUE" ]] || die "queue does not exist: $QUEUE"
  local total claimed done_n failed_n
  total=$(find "$QUEUE/tasks" -mindepth 1 -maxdepth 1 -type f -name '*.task' | wc -l)
  claimed=$(find "$QUEUE/claims" -mindepth 1 -maxdepth 1 -type d | wc -l)
  done_n=$(find "$QUEUE/done" -mindepth 1 -maxdepth 1 -type f | wc -l)
  failed_n=$(find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f | wc -l)
  echo "STATUS total=$total claimed=$claimed validated=$done_n failed=$failed_n"
  echo "IN_FLIGHT_OR_STALE=$((claimed - done_n - failed_n))"
  if [[ -f "$QUEUE/FINALIZED" ]]; then
    echo "FINALIZED"
    cat "$QUEUE/FINALIZED"
    cat "$FINAL_ROOT/tag10_5cef834_3seed_8task.final_row.txt"
  fi
  if [[ "$failed_n" -gt 0 ]]; then
    echo "FAILED_TASKS"
    find "$QUEUE/failed" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort
  fi
  echo "NODE_STATE"
  find "$QUEUE/nodes" -maxdepth 1 -type f -printf '%f\n' | sort
  echo "RECENT_TASK_LOGS"
  for f in "$LOG_ROOT"/tasks/*.log; do
    [[ -e "$f" ]] || continue
    printf '%s  ' "$(basename "$f")"
    tail -n 1 "$f"
  done
}

case "${1:-}" in
  --prepare) prepare_queue ;;
  --node-worker) node_worker ;;
  --status) show_status ;;
  "") launch_node ;;
  *) die "usage: bash $0 [--prepare|--status]" ;;
esac
