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
alpaca-cleaned  alpaca-gpt4   alpaca_gpt4     apps        bbh
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

### Two Alpaca-GPT4 mirrors — they are not interchangeable

`alpaca_gpt4/` is **liangxin/Alpaca_GPT4**, which `configs/base.yaml` has
always named as its `dataset_name` fallback, so it is the mirror the earlier
runs drew from and the one Table 2 must use.

`alpaca-gpt4/` is **vicgalle/alpaca-gpt4** — same corpus, but it ships an
extra pre-formatted `text` column and twice the bytes. Using it changes the
tokenised text.

A table row cannot be reproduced without saying which mirror produced it.
Cite the sha256 (`scripts/export_hf_corpus.py --inspect`), not the name.

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
  existing. It has caught a corpus one level down more than once.
- `python scripts/gate_report.py --gate <cache>` — the gate's G distribution
  and, where corruption labels exist, its separation. Costs seconds and
  answers whether the training run is worth starting.
- `python scripts/selection_purity.py` — what each arm actually trained on.
  Lands epochs before any eval and is more diagnostic than one.
- The forward-batch size has silently fallen back to the small-GPU default
  twice, from a shell that did not export it. `env.sh` prints
  `fwd batch : 0.5b=… 7b=…`; check it.
