#!/usr/bin/env python3
"""
Expand seed AppleScript pairs into a larger synthetic training dataset
using the Anthropic API (Claude).

Reads data/seed_pairs.jsonl, generates 50 variations per seed pair, and
saves results to data/expanded_pairs.jsonl.

Supports resuming: if expanded_pairs.jsonl already has enough lines, the
script skips generation. If partially complete, it continues from where
it left off.

Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_PATH = DATA_DIR / "seed_pairs.jsonl"
EXPANDED_PATH = DATA_DIR / "expanded_pairs.jsonl"

VARIATIONS_PER_SEED = 50
BATCH_SIZE = 2  # seed pairs per API call (2 seeds x 50 variations = 100 items per call)
RATE_LIMIT_DELAY = 2.0  # seconds between API calls
TARGET_MIN_PAIRS = 10_000
TARGET_MAX_PAIRS = 20_000
MAX_TOKENS = 8192  # enough room for ~100 JSON objects per API call
MODEL = "claude-sonnet-4-20250514"


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_completed_seed_indices(expanded_path: Path) -> set[int]:
    """
    Read expanded_pairs.jsonl and return the set of seed_index values
    that have already been processed.
    """
    indices = set()
    records = load_jsonl(expanded_path)
    for rec in records:
        if "seed_index" in rec:
            indices.add(rec["seed_index"])
    return indices


def build_prompt(seed_batch: list[dict]) -> str:
    """
    Build a prompt asking Claude to generate variations for a batch of
    seed pairs.
    """
    examples_block = ""
    for i, pair in enumerate(seed_batch):
        examples_block += f"""
--- Seed {i+1} ---
Input: {pair['input']}
Output: {pair['output']}
"""

    prompt = f"""You are helping create training data for a small language model that converts natural language Mac commands into AppleScript code.

Below are seed examples. For EACH seed, generate exactly {VARIATIONS_PER_SEED} variations. Each variation should:

1. Rephrase the natural language input in a different way (casual, formal, terse, verbose, different word choices, synonyms, different sentence structures). Make them diverse -- some short commands, some full sentences, some questions like "how do I...", some imperative like "please...", etc.
2. Provide the correct corresponding AppleScript output. The AppleScript should accomplish the same task but may vary slightly in style (e.g., using "activate" vs not, different variable names, using "set" vs direct commands where both are valid). Most variations should produce functionally equivalent AppleScript but with natural variation.

{examples_block}

Return your response as a JSON array of objects. Each object must have exactly these fields:
- "input": the natural language command (string)
- "output": the AppleScript code (string)
- "seed_index": which seed number this is a variation of (integer, 1-indexed matching the seed numbers above)

Return ONLY the JSON array, no other text. Make sure the JSON is valid.
Example format:
[
  {{"input": "...", "output": "...", "seed_index": 1}},
  {{"input": "...", "output": "...", "seed_index": 1}},
  ...
]"""
    return prompt


def parse_response(text: str) -> list[dict]:
    """
    Parse Claude's response into a list of {input, output, seed_index} dicts.
    Handles cases where response has markdown code fences.
    """
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON response: {e}")
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                print("  Error: Could not extract valid JSON from response")
                return []
        else:
            return []

    if not isinstance(data, list):
        print("  Warning: Response is not a JSON array")
        return []

    valid = []
    for item in data:
        if isinstance(item, dict) and "input" in item and "output" in item:
            valid.append({
                "input": str(item["input"]),
                "output": str(item["output"]),
                "seed_index": item.get("seed_index", 0),
            })
    return valid


def expand_batch(
    client: anthropic.Anthropic,
    seed_batch: list[dict],
    global_seed_indices: list[int],
) -> list[dict]:
    """
    Call the Anthropic API to expand a batch of seed pairs.
    Returns list of variation dicts with corrected seed_index mapped to
    global indices.
    """
    prompt = build_prompt(seed_batch)

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.RateLimitError:
        print("  Rate limited, waiting 60 seconds...")
        time.sleep(60)
        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        print(f"  API error: {e}")
        return []

    response_text = message.content[0].text
    variations = parse_response(response_text)

    # Remap seed_index from batch-local (1-indexed) to global
    remapped = []
    for var in variations:
        local_idx = var.get("seed_index", 1)
        # Clamp to valid range
        local_idx = max(1, min(local_idx, len(seed_batch)))
        global_idx = global_seed_indices[local_idx - 1]
        remapped.append({
            "input": var["input"],
            "output": var["output"],
            "seed_index": global_idx,
        })

    return remapped


def main():
    # ---- Check API key ----
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    # ---- Load seed data ----
    seeds = load_jsonl(SEED_PATH)
    if not seeds:
        print(f"Error: No seed data found at {SEED_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(seeds)} seed pairs from {SEED_PATH}")

    # ---- Check if we already have enough expanded data ----
    existing_count = count_lines(EXPANDED_PATH)
    expected_total = len(seeds) * VARIATIONS_PER_SEED
    print(f"Existing expanded pairs: {existing_count}")
    print(f"Expected total: {expected_total} ({len(seeds)} seeds x {VARIATIONS_PER_SEED} variations)")

    if existing_count >= TARGET_MIN_PAIRS:
        print(f"Already have {existing_count} pairs (target minimum: {TARGET_MIN_PAIRS}). Skipping.")
        return

    # ---- Determine which seeds still need processing ----
    completed_indices = get_completed_seed_indices(EXPANDED_PATH)
    remaining_indices = [
        i for i in range(len(seeds)) if i not in completed_indices
    ]
    print(f"Seeds already processed: {len(completed_indices)}")
    print(f"Seeds remaining: {len(remaining_indices)}")

    if not remaining_indices:
        print("All seeds have been processed.")
        return

    # ---- Process in batches ----
    client = anthropic.Anthropic(api_key=api_key)
    total_generated = existing_count

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
        if total_generated >= TARGET_MAX_PAIRS:
            print(f"Reached target maximum of {TARGET_MAX_PAIRS} pairs. Stopping.")
            break

        batch_indices = remaining_indices[batch_start:batch_start + BATCH_SIZE]
        batch_seeds = [seeds[i] for i in batch_indices]

        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(remaining_indices) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches}: Processing seeds {batch_indices}")

        variations = expand_batch(client, batch_seeds, batch_indices)

        if variations:
            # Append to file
            with open(EXPANDED_PATH, "a", encoding="utf-8") as f:
                for var in variations:
                    f.write(json.dumps(var, ensure_ascii=False) + "\n")

            total_generated += len(variations)
            print(f"  Generated {len(variations)} variations. Total: {total_generated}")
        else:
            print("  No variations generated for this batch.")

        # Rate limit delay
        time.sleep(RATE_LIMIT_DELAY)

    print(f"\nDone. Total expanded pairs: {total_generated}")
    print(f"Saved to {EXPANDED_PATH}")


if __name__ == "__main__":
    main()
