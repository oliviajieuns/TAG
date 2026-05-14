"""Merge a LoRA adapter into its base model and save the result.

Usage:
    python scripts/merge_lora.py \\
        --base_model /path/to/base \\
        --adapter_path /path/to/lora_ckpt \\
        --output_path /path/to/merged
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_path", required=True)
    return p.parse_args()


def _from_pretrained(path: str, cls):
    local = os.path.exists(path)
    return cls.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if cls is AutoModelForCausalLM else None,
        trust_remote_code=True,
        local_files_only=local,
    )


def main() -> None:
    args = parse_args()

    print(f"Loading base model from {args.base_model}")
    base = _from_pretrained(args.base_model, AutoModelForCausalLM)

    print(f"Loading LoRA adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging LoRA weights ...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}")
    merged.save_pretrained(args.output_path)

    print("Saving tokenizer ...")
    tokenizer = _from_pretrained(args.base_model, AutoTokenizer)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
