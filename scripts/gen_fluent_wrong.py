#!/usr/bin/env python
"""Generate fluent-but-wrong (T7) replacement responses on the GPU server.

Step 2 of the T7 workflow in ``scripts/make_corrupted_pool.py``: reads the
targets file emitted by ``--emit-fluent-wrong-targets`` (a JSON list of
``{"index", "instruction", "input", "target_words"}`` entries), prompts an
instruct model to write a fluent, on-topic but subtly INCORRECT or
confidently vacuous response for each target, and writes

    {"<pool_index>": "<replacement response>", ...}

for ``make_corrupted_pool.py --fluent-wrong-file``.

This is research tooling for building a LABELED robustness benchmark:
every generated response is recorded with a ground-truth ``fluent_wrong``
manifest entry, so the texts exist solely to measure whether data-quality
filters (perplexity, counterfactual, MVF) catch fluent-but-wrong training
data. Nothing here is intended for, or usable as, deployed deceptive
content — outputs are keyed to a corruption manifest by construction.

Determinism: targets are processed in sorted-index order, the torch RNG is
re-seeded per batch from ``--seed`` + batch start, and sampling parameters
are fixed, so output is deterministic for a fixed (seed, batch-size).
Changing ``--batch-size`` regroups the targets into different batches and
DOES change the sampled texts — pin the batch size along with the seed.
Reruns with the same flags on the same hardware/library versions reproduce
the same file.

Heavy imports (torch/transformers) happen inside the generation path only,
so ``--dry-run`` — which emits deterministic placeholder texts of roughly
``target_words`` length — works on machines without them and lets the
downstream plumbing be tested end-to-end without a model.

Examples:
    python scripts/gen_fluent_wrong.py \
        --targets data/lowq/composite20/fluent_wrong_targets.json \
        --out data/lowq/composite20/fluent_wrong.json \
        --model Qwen/Qwen2.5-7B-Instruct --batch-size 8

    # plumbing test, no model:
    python scripts/gen_fluent_wrong.py \
        --targets .../fluent_wrong_targets.json --out fw.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

_SYSTEM_PROMPT = (
    "You write test items for a data-quality robustness benchmark. Each "
    "answer you produce is stored with a ground-truth 'corrupted' label "
    "and is used only to evaluate automatic data filters."
)

_USER_TEMPLATE = (
    "Task instruction:\n{instruction}\n"
    "{input_block}"
    "\nWrite a response of about {target_words} words that reads as fluent, "
    "confident and on-topic but is WRONG: either it contains a subtle "
    "factual or logical error, or it is confidently vacuous (asserts a lot "
    "while never actually answering the task). Do NOT include disclaimers, "
    "hedges, meta-commentary, or any hint that the response is flawed. "
    "Output only the response text."
)


def _load_targets(path: str, limit: int | None) -> List[Dict[str, Any]]:
    # encoding pinned: instructions are UTF-8 JSON and the process default
    # on Windows is cp949 — locale-dependent decode crashes are unacceptable.
    with open(path, encoding="utf-8") as f:
        targets = json.load(f)
    if not isinstance(targets, list):
        raise ValueError(f"{path}: expected a JSON list of target entries")
    for i, t in enumerate(targets[:5]):
        if "index" not in t or "instruction" not in t:
            raise ValueError(
                f"{path}: entry {i} missing 'index'/'instruction' keys "
                f"(got {sorted(t)})"
            )
    targets = sorted(targets, key=lambda t: int(t["index"]))
    return targets[:limit] if limit is not None else targets


def _dump(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ensure_ascii=False emits raw unicode, so the encoding must be pinned
    # (Windows defaults to cp949, which cannot encode arbitrary model text).
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=None)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"wrote {path}")


def _user_prompt(target: Dict[str, Any]) -> str:
    inp = str(target.get("input", "") or "")
    input_block = f"Task input:\n{inp}\n" if inp.strip() else ""
    return _USER_TEMPLATE.format(
        instruction=str(target.get("instruction", "")),
        input_block=input_block,
        target_words=max(8, int(target.get("target_words", 40))),
    )


def _dry_run_text(target: Dict[str, Any]) -> str:
    """Deterministic placeholder of roughly target_words words; keeps the
    length-bucket statistics of a real run so downstream plumbing (and
    length-bias checks) behave the same without a model."""
    n_words = max(8, int(target.get("target_words", 40)))
    filler = (
        "It is widely accepted that the answer follows directly from first "
        "principles and standard results in the field."
    ).split()
    words = [f"[dry-run-fluent-wrong-{int(target['index'])}]"]
    words += [filler[k % len(filler)] for k in range(n_words - 1)]
    return " ".join(words)


def _generate(targets: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, str]:
    # Heavy imports live here so --dry-run needs neither torch nor
    # transformers installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto",
    )
    model.eval()

    out: Dict[str, str] = {}
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start:start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(t)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in batch
        ]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_prompt_tokens,
        ).to(model.device)
        # Re-seed per batch from the batch's absolute start position: reruns
        # with a fixed (seed, batch-size) reproduce the same texts. NOTE the
        # batch size is part of the determinism key — a different
        # --batch-size regroups targets into different batches (and the RNG
        # stream interleaves across a batch), changing the output.
        torch.manual_seed(args.seed + start)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for t, row in zip(batch, gen):
            text = tokenizer.decode(
                row[prompt_len:], skip_special_tokens=True,
            ).strip()
            out[str(int(t["index"]))] = text
        print(f"generated {min(start + len(batch), len(targets))}/{len(targets)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--targets", required=True,
                   help="fluent_wrong_targets.json from make_corrupted_pool.py")
    p.add_argument("--out", required=True,
                   help='output JSON {"<pool_index>": "<response>", ...}')
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N targets (by index)")
    p.add_argument("--dry-run", action="store_true",
                   help="no model load; emit deterministic placeholder texts "
                        "so the downstream plumbing is testable")
    args = p.parse_args()

    targets = _load_targets(args.targets, args.limit)
    if not targets:
        raise SystemExit(f"{args.targets}: no targets to process")

    if args.dry_run:
        out = {str(int(t["index"])): _dry_run_text(t) for t in targets}
    else:
        out = _generate(targets, args)

    _dump(out, Path(args.out))
    print(f"T7 replacements for {len(out)} targets "
          f"({'dry-run placeholders' if args.dry_run else args.model})")


if __name__ == "__main__":
    main()
