# TAG — working notes

## Datasets live in exactly one place

**`/group-volume/datasets/<corpus>/`** on the n9 cluster. Every corpus and
benchmark was consolidated there (2026-08); the HF `datasets` arrow caches
under `/group-volume/data/hf_home/datasets` were materialised as parquet and
deleted. There is no second location.

If something is not there, it has to be downloaded — do not go looking for it
under `HF_HOME`, `/group-volume/IT-datasets`, or a per-user directory. Those
paths still appear in older scripts as fallbacks; they are historical.

Present as of the consolidation:

```
alpaca-cleaned  alpaca-gpt4   apps            bbh
coconot         dapo-math-17k fineweb         gsm8k       hendrycks_math
hh-rlhf         humaneval     kk              knights-and-knaves
knowledge-pile  math500       mbpp            mmlu        muse-books
muse-news       openwebmath   rwku            svamp       tofu
tulu-3-sft-mixture            tydiqa          ultrafeedback_binarized
wildjailbreak   wmdp          xquad
```

Layout inside a corpus is `<split>-00000-of-00001.parquet`, or
`<config>/<split>-*.parquet` when the dataset has more than one config
(`gsm8k/main/test.parquet`, `rwku/forget_level2/train-*.parquet`). Anything
this repo wrote carries a `SOURCE.json` naming the cache entry it came from
and the row count per file.

### Alpaca-GPT4 is `alpaca-gpt4/` — and it is the vicgalle mirror

One copy is kept: **`/group-volume/datasets/alpaca-gpt4`**, which is
`vicgalle/alpaca-gpt4`. `configs/base.yaml` names `liangxin/Alpaca_GPT4` as
its `dataset_name` fallback, so the corpus in use is NOT the one the config
file names. The two differ by a pre-formatted `text` column, which
`make_corrupted_pool.py` drops, and possibly by record count, which it does
not — check `n_total` in the pool manifest against the 52 002 the paper
assumes.

A table row cannot be reproduced without saying which mirror produced it.
Cite the sha256, not the name — `make_corrupted_pool.py` records it for the
corpus a pool was built from, under `inputs` in that pool's
`corruption_manifest.json`.

The Table 2 pool (`$POOLS/alpaca_gpt4`, built 2026-08, seed 42) came from:

```
vicgalle/alpaca-gpt4   /group-volume/datasets/alpaca-gpt4
train-00000-of-00001.parquet
sha256 7f16a6f433119e28a9ff906cdb74752c28a721dffd1f6e45600a1d90e57f2543
n_total 52002
```

## Two experiments that are easy to confuse

| | Table 2 row | lowq robustness grid |
|---|---|---|
| backbone | LLaMA-2-7B | Qwen2.5-7B-**Instruct** |
| corpus | Alpaca-GPT4, clean | composite20, 30.4 % corrupted |
| configs | `configs/experiments/main_7b/llama2/` | `configs/experiments/lowq/` |
| launcher | `scripts/run_main_7b.sh` (DDP ×4) | `scripts/run_lowq_all_arms.sh` (1 arm/GPU) |

`scripts/gpu_cloud/env.sh` points `ALPACA_DATA_FILES` at the **corrupted**
composite20 pool, because that is what the lowq grid selects from. A Table 2
row that inherited it would train on corrupted data and report a number
nobody could explain, so `main_7b/llama2/tag_10.yaml` pins its pool to
`TAG_MAIN_POOL` instead. Run `scripts/check_row_pair.py` before launching a
pair — it refuses a pool that looks corrupted and reports every config key
the two rows differ on.

## Before spending GPU hours

- `python scripts/check_eval_data.py` — the eight Table 2 benchmarks, checked
  against the files each evaluator actually opens, not against the directory
  existing. It has caught a corpus one level down more than once, and an
  MMLU that resolved to a single subject.
  Benchmarks are acquired by `scripts/download_<bench>.sh` (six of the
  eight); MMLU and BBH have no downloader and are built from the clones on
  disk by `scripts/prepare_eval_data.py --apply`.
- `python scripts/gate_report.py --gate <cache>` — the gate's G distribution
  and, where corruption labels exist, its separation. Costs seconds and
  answers whether the training run is worth starting.
- `python scripts/selection_purity.py` — what each arm actually trained on.
  Lands epochs before any eval and is more diagnostic than one.
- The forward-batch size has silently fallen back to the small-GPU default
  twice, from a shell that did not export it. `env.sh` prints
  `fwd batch : 0.5b=… 7b=…`; check it.
