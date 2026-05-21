"""AlpaGasus (Chen et al., 2024) — pre-filtered Alpaca subset SFT.

The official repo (github.com/gpt4life/alpagasus) ships pre-computed
GPT-4-rated selections (`data/filtered/chatgpt_9k.json`, threshold 4.5),
so we don't have to re-run any OpenAI API rating. This subpackage matches
the filtered instruction strings against our local Alpaca-GPT4 records and
SFTs on the matched subset.
"""
