#!/usr/bin/env bash
# Download XQuAD — all 11 languages.
#
# Usage:
#   bash scripts/download_xquad.sh [target_dir]
#
# If target_dir is omitted, falls back to
#   ${XQUAD_DATA_DIR:-/group-volume/IT-datasets/xquad}
#
# After completion the directory layout will be:
#   <target_dir>/
#     xquad.ar.json
#     xquad.de.json
#     xquad.el.json
#     xquad.en.json
#     xquad.es.json
#     xquad.hi.json
#     xquad.ro.json
#     xquad.ru.json
#     xquad.th.json
#     xquad.tr.json
#     xquad.vi.json
#     xquad.zh.json
#
# (12 languages — NAIT Table 2 reports XQuAD as macro-avg over 11. The extra
# one is `ro` (Romanian) which Google added later; our evaluator includes
# all 12 by default and reports per-language. To match NAIT exactly drop
# `ro` in the evaluator's `--languages` arg.)
#
# Source: the official google/xquad GitHub repo:
#   https://github.com/google-deepmind/xquad
# Each language file is the SQuAD-v1 JSON shape:
#   { "data": [ { "paragraphs": [ { "context": ..., "qas": [...] } ] } ] }
set -euo pipefail

TARGET="${1:-${XQUAD_DATA_DIR:-/group-volume/IT-datasets/xquad}}"

parent=$(dirname "$TARGET")
if [ ! -d "$parent" ]; then
  echo "[error] parent directory does not exist: $parent" >&2
  exit 2
fi
if ! mkdir -p "$TARGET" 2>/dev/null; then
  echo "[error] cannot create target directory: $TARGET" >&2
  ls -ld "$parent" >&2 || true
  exit 2
fi
if [ ! -w "$TARGET" ]; then
  echo "[error] target directory is not writable: $TARGET" >&2
  ls -ld "$TARGET" >&2 || true
  exit 2
fi

# 12 languages — NAIT 11 + ro (extra). Per-paper comparison drops `ro` at
# evaluator level via --languages.
LANGS=(ar de el en es hi ro ru th tr vi zh)

# Two URL roots — official GitHub mirror first, HF mirror fallback.
GH_ROOT="https://raw.githubusercontent.com/google-deepmind/xquad/master"
HF_ROOT="https://huggingface.co/datasets/google/xquad/resolve/main"

# Connectivity probe — try English first (smallest payload that's always present).
_probe_ok=0
for url in "$GH_ROOT/xquad.en.json" "$HF_ROOT/xquad.en.json"; do
  if curl -fLsS -o /dev/null --max-time 10 -I "$url" 2>/dev/null; then
    _probe_ok=1
    echo "[probe] reachable via: $url"
    break
  fi
done
if [ "$_probe_ok" = "0" ]; then
  echo "[error] cannot reach any XQuAD JSON URL." >&2
  echo "        Tried:" >&2
  echo "          $GH_ROOT/xquad.en.json" >&2
  echo "          $HF_ROOT/xquad.en.json" >&2
  echo "        Manual fallback (any node with HTTPS):" >&2
  echo "          for lang in ${LANGS[*]}; do" >&2
  echo "            curl -fL $GH_ROOT/xquad.\$lang.json -o $TARGET/xquad.\$lang.json" >&2
  echo "          done" >&2
  exit 3
fi

fetch_one() {
  local lang="$1"
  local dst="$TARGET/xquad.$lang.json"
  if [ -s "$dst" ]; then
    echo "[skip] xquad.$lang.json already present ($(stat -c%s "$dst") bytes)"
    return 0
  fi
  for root in "$GH_ROOT" "$HF_ROOT"; do
    local url="$root/xquad.$lang.json"
    echo "  [try] $url"
    if curl -fL --retry 3 --retry-delay 2 -o "$dst.part" "$url" 2>&1 | tail -2; then
      mv "$dst.part" "$dst"
      echo "  [ok]  xquad.$lang.json  →  $(stat -c%s "$dst") bytes"
      return 0
    fi
    rm -f "$dst.part"
    echo "  [fail] $url"
  done
  echo "[error] every URL failed for $lang" >&2
  return 1
}

# Romanian (`ro`) is the only language that's intermittently missing from
# the github mirror — make it non-fatal so the user doesn't lose the other
# 11 if Google removes / re-adds it.
fail=0
for lang in "${LANGS[@]}"; do
  if ! fetch_one "$lang"; then
    if [ "$lang" = "ro" ]; then
      echo "[warn] Romanian (ro) fetch failed — continuing (non-NAIT)" >&2
    else
      fail=1
    fi
  fi
done
if [ "$fail" = "1" ]; then
  exit 1
fi

echo ""
echo "XQuAD ready at: $TARGET"
n_files=$(ls -1 "$TARGET"/xquad.*.json 2>/dev/null | wc -l)
echo "  Languages downloaded: $n_files"
echo ""
echo "If XQUAD_DATA_DIR is not already set in scripts/setup_env.sh, export it:"
echo "    export XQUAD_DATA_DIR=\"$TARGET\""
echo ""
echo "Sanity check:"
echo "    python3 -c \"import json; d=json.load(open('$TARGET/xquad.en.json')); print('en paragraphs:', sum(len(a['paragraphs']) for a in d['data']))\""
