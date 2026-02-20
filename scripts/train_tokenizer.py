#!/usr/bin/env python3
"""
Train a BPE tokenizer on seed and expanded AppleScript data.

Reads data/seed_pairs.jsonl and data/expanded_pairs.jsonl (if it exists),
trains a BPE tokenizer with vocab_size=8192, adds special tokens, and saves
to model/tokenizer.json.
"""

import json
import os
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

SPECIAL_TOKENS = ["<|input|>", "<|output|>", "<|end|>", "<|pad|>"]
VOCAB_SIZE = 8192


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_num} in {path}: {e}")
    return records


def collect_texts(records: list[dict]) -> list[str]:
    """Extract all input and output text from records."""
    texts = []
    for rec in records:
        if "input" in rec:
            texts.append(rec["input"])
        if "output" in rec:
            texts.append(rec["output"])
    return texts


def main():
    # ---- Gather training texts ----
    seed_path = DATA_DIR / "seed_pairs.jsonl"
    expanded_path = DATA_DIR / "expanded_pairs.jsonl"

    seed_records = load_jsonl(seed_path)
    expanded_records = load_jsonl(expanded_path)

    print(f"Loaded {len(seed_records)} seed pairs from {seed_path}")
    print(f"Loaded {len(expanded_records)} expanded pairs from {expanded_path}")

    all_records = seed_records + expanded_records
    if not all_records:
        print("Error: No training data found. Place JSONL files in data/", file=sys.stderr)
        sys.exit(1)

    texts = collect_texts(all_records)
    print(f"Total text segments for tokenizer training: {len(texts)}")

    # ---- Also include the special-token formatted sequences so the tokenizer
    #      sees them in context ----
    for rec in all_records:
        formatted = f"<|input|> {rec.get('input', '')} <|output|> {rec.get('output', '')} <|end|>"
        texts.append(formatted)

    # ---- Build BPE tokenizer ----
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    # Train from the in-memory iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # ---- Post-processing: nothing fancy, just ensure the special tokens are
    #      accessible by ID ----
    # The trainer already reserves slots for special tokens at IDs 0-3.

    # ---- Save ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"\nTokenizer saved to {out_path}")

    # ---- Print summary ----
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    # Show special token IDs
    print("\nSpecial token IDs:")
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        print(f"  {tok} -> {tid}")

    # Sample encode / decode
    sample_input = "open Safari and go to google.com"
    sample_output = 'tell application "Safari" to open location "https://google.com"'
    sample_formatted = f"<|input|> {sample_input} <|output|> {sample_output} <|end|>"

    encoded = tokenizer.encode(sample_formatted)
    print(f"\nSample text:\n  {sample_formatted}")
    print(f"Encoded IDs ({len(encoded.ids)} tokens):\n  {encoded.ids}")
    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded back:\n  {decoded}")


if __name__ == "__main__":
    main()
