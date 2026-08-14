#!/usr/bin/env bash
# Find what this cluster already has, and write a sourceable env file.
#
#   bash scripts/gpu_cloud/n9_discover.sh            # probe + print
#   bash scripts/gpu_cloud/n9_discover.sh --write    # ...and save it
#
# Writes $TAG_ROOT/discovered_env.sh, which n9_env.sh sources automatically
# if present. Run this ONCE on a new box instead of hand-exporting paths.
#
# It probes for four things and never guesses silently — anything not found
# is reported as MISSING with what to do about it:
#   1. model checkpoints (a directory containing config.json)
#   2. an instruction corpus in Alpaca schema (instruction/input/output)
#   3. an HF datasets cache (hf_home/datasets/<org>___<name>)
#   4. per-benchmark eval directories
set -uo pipefail

WRITE=0
[ "${1:-}" = "--write" ] && WRITE=1

TAG_ROOT="${TAG_ROOT:-/group-volume/jieuns.shin/tag/tests/tag}"
OUT="$TAG_ROOT/discovered_env.sh"

# Roots to search, cheapest first. Add your own by exporting EXTRA_ROOTS.
MODEL_ROOTS="/group-volume/models /group-volume/nait-models /group-volume/data/models ${EXTRA_MODEL_ROOTS:-}"
# $TAG_WORKSPACE first: bootstrap materialises Alpaca-GPT4 there, and that
# copy should win over whatever else happens to be lying around.
DATA_ROOTS="${TAG_WORKSPACE:-$TAG_ROOT/workspace}/datasets /group-volume/${USER:-nobody}/datasets /group-volume/datasets /group-volume/IT-datasets /group-volume/data/datasets /group-volume/data ${EXTRA_DATA_ROOTS:-}"
HF_ROOTS="/group-volume/data/hf_home /group-volume/hf_home ${EXTRA_HF_ROOTS:-}"

say()  { printf '%s\n' "$*"; }
kv()   { printf '  %-22s %s\n' "$1" "$2"; }

declare -a EXPORTS=()
add_export() { EXPORTS+=("export $1=\"$2\""); }

say "=============================================================="
say " TAG cluster discovery"
say "=============================================================="

# ---------------------------------------------------------------- models
say ""
say "MODELS (dirs containing config.json)"
find_model() {  # find_model <regex> ; echoes first match
  local pat="$1" r
  for r in $MODEL_ROOTS; do
    [ -d "$r" ] || continue
    # -maxdepth 2 so <root>/<name>/config.json and <root>/<org>/<name>/ both hit
    local hit
    hit="$(find "$r" -maxdepth 3 -name config.json -path "*${pat}*" 2>/dev/null | head -1)"
    [ -n "$hit" ] && { dirname "$hit"; return; }
  done
  echo ""
}
M7B="$(find_model 'Qwen2.5-7B')"
M7B_BASE="$(find_model 'Qwen2.5-7B' | grep -iv instruct || true)"
M05B="$(find_model 'Qwen2.5-0.5B')"
for r in $MODEL_ROOTS; do [ -d "$r" ] && kv "root" "$r"; done
if [ -n "$M7B" ]; then
  kv "Qwen2.5-7B" "$M7B"
  add_export MODEL_PATH_QWEN25_7B "$M7B"
  case "$M7B" in *[Ii]nstruct*)
    say "     ^ INSTRUCT, not base. Not a neutral substitution: the paper's"
    say "       setup is SFT from a base checkpoint, AND an instruction-tuned"
    say "       model separates clean from corrupted more easily, so the gate"
    say "       looks better than base would show. State it, or fetch base."
  ;; esac
else
  kv "Qwen2.5-7B" "MISSING -> bash scripts/gpu_cloud/bootstrap.sh model7b"
fi
[ -n "$M05B" ] && { kv "Qwen2.5-0.5B" "$M05B"; add_export MODEL_PATH_QWEN25_05B "$M05B"; } \
               || kv "Qwen2.5-0.5B" "MISSING -> bootstrap.sh model"

# ------------------------------------------------------- instruction data
say ""
say "INSTRUCTION CORPUS (Alpaca schema: instruction / input / output)"
CORPUS=""
CANDIDATES=()
for r in $DATA_ROOTS; do
  [ -d "$r" ] || continue
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if python - "$f" <<'PY' >/dev/null 2>&1
import json, sys
recs = json.load(open(sys.argv[1]))
assert isinstance(recs, list) and len(recs) > 100
r = recs[0]
assert "instruction" in r and "output" in r
PY
    then CANDIDATES+=("$f"); fi
  # -size +100k, not +1M: a real Alpaca dump is tens of MB, but a subset or a
  # jsonl-converted variant can be much smaller and is still valid input.
  done < <(find "$r" -maxdepth 6 -name '*alpaca*.json' -size +100k 2>/dev/null | head -20)
done
# Prefer Alpaca-GPT4 when several corpora exist: it is what base.yaml and the
# prior legacy-score numbers use, so silently picking another would break
# comparability without anyone noticing.
for c in ${CANDIDATES[@]+"${CANDIDATES[@]}"}; do
  case "$c" in *gpt4*|*GPT4*) CORPUS="$c"; break ;; esac
done
if [ -z "$CORPUS" ] && [ ${#CANDIDATES[@]} -gt 0 ]; then CORPUS="${CANDIDATES[0]}"; fi

# An HF-cached corpus is not a .json on disk, but it is just as usable
# offline — and Alpaca-GPT4 is usually only present in that form.
HF_ALPACA_GPT4=""
for r in $HF_ROOTS; do
  [ -d "$r/datasets" ] || continue
  # '?' matches the one char that differs between the alpaca-gpt4 and
  # alpaca_gpt4 mirrors.
  if [ -n "$(find "$r/datasets" -maxdepth 1 -type d -iname '*alpaca?gpt4*' 2>/dev/null | head -1)" ]; then
    HF_ALPACA_GPT4="$r"; break
  fi
done

if [ -n "$CORPUS" ]; then
  N="$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$CORPUS" 2>/dev/null)"
  kv "corpus" "$CORPUS"
  kv "records" "$N"
  add_export ALPACA_RAW_JSON "$CORPUS"
  case "$CORPUS" in
    *gpt4*|*GPT4*) : ;;
    *)
      say "     ^ this is NOT alpaca-GPT4, which is what base.yaml and the"
      say "       prior legacy-score numbers use — results will not be comparable."
      if [ -n "$HF_ALPACA_GPT4" ]; then
        say "       BUT alpaca-gpt4 IS cached at $HF_ALPACA_GPT4/datasets."
        say "       Materialise it (offline, no download) and prefer it:"
        say "         bash scripts/gpu_cloud/bootstrap.sh data"
        say "         bash scripts/gpu_cloud/n9_discover.sh --write   # re-detect"
      fi
    ;;
  esac
  case "$CORPUS" in
    *cleaned*)
      say "       (alpaca-cleaned is a defensible base for THIS experiment —"
      say "        the corruption manifest assumes a clean starting pool and"
      say "        this one explicitly is — but pre-register the choice.)"
    ;;
  esac
else
  kv "corpus" "MISSING -> bash scripts/gpu_cloud/bootstrap.sh data"
  [ -n "$HF_ALPACA_GPT4" ] && \
    say "     (alpaca-gpt4 is cached at $HF_ALPACA_GPT4 — that step will use it offline)"
fi

# ------------------------------------------------------------- HF cache
say ""
say "HF DATASETS CACHE"
HFHOME=""
for r in $HF_ROOTS; do
  if [ -d "$r/datasets" ] || [ -d "$r/hub" ]; then HFHOME="$r"; break; fi
done
if [ -n "$HFHOME" ]; then
  kv "HF_HOME" "$HFHOME"
  add_export HF_HOME "$HFHOME"
  # -type d and no .lock: the cache dir is littered with lock FILES whose
  # names also contain '___', which would otherwise dominate the listing.
  n="$(find "$HFHOME/datasets" -maxdepth 1 -type d -name '*___*' 2>/dev/null | wc -l)"
  kv "cached datasets" "$n"
  find "$HFHOME/datasets" -maxdepth 1 -type d -name '*___*' 2>/dev/null \
    | sed 's|.*/|       - |' | sort | head -15
else
  kv "HF_HOME" "MISSING (only needed for eval benchmarks)"
fi

# ------------------------------------------------------------ benchmarks
say ""
say "EVAL BENCHMARK DIRS"
probe_bench() {  # probe_bench <ENVVAR> <dirname-pattern>
  local var="$1" pat="$2" r hit
  for r in $DATA_ROOTS; do
    [ -d "$r" ] || continue
    hit="$(find "$r" -maxdepth 3 -type d -iname "$pat" 2>/dev/null | head -1)"
    [ -n "$hit" ] && { kv "$var" "$hit"; add_export "$var" "$hit"; return; }
  done
  kv "$var" "-"
}
probe_bench MMLU_DATA_DIR      'mmlu'
probe_bench MMLU_PRO_DATA_DIR  'mmlu_pro'
probe_bench GSM8K_DATA_DIR     'gsm8k'
probe_bench SVAMP_DATA_DIR     'svamp'
probe_bench HUMANEVAL_DATA_DIR 'human*eval'
probe_bench MBPP_DATA_DIR      'mbpp'
probe_bench TYDIQA_DATA_DIR    'tydiqa'
probe_bench XQUAD_DATA_DIR     'xquad'
probe_bench BBH_DATA_DIR       'bbh'
say '  (eval dirs are only needed for "python -m tag.eval", not for training)'

# ----------------------------------------------------------------- emit
say ""
say "=============================================================="
if [ ${#EXPORTS[@]} -eq 0 ]; then
  say " nothing found — check the roots at the top of this script"
  exit 1
fi
say " ENV BLOCK"
say "=============================================================="
printf '%s\n' "${EXPORTS[@]}"

if [ "$WRITE" -eq 1 ]; then
  mkdir -p "$TAG_ROOT"
  {
    echo "# Generated by scripts/gpu_cloud/n9_discover.sh"
    echo "# Re-run that script to refresh. n9_env.sh sources this if present."
    printf '%s\n' "${EXPORTS[@]}"
  } > "$OUT"
  say ""
  say "written to $OUT"
  say "n9_env.sh will pick it up automatically."
else
  say ""
  say "(re-run with --write to save it to $OUT)"
fi
