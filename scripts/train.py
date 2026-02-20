#!/usr/bin/env python3
"""
Train a decoder-only transformer to convert natural language Mac commands
into AppleScript, using MLX on Apple Silicon.

Loads the tokenizer from model/tokenizer.json, reads JSONL data from data/,
tokenizes into the format:
    <|input|> {input} <|output|> {output} <|end|>
and trains with cross-entropy loss masked to only supervise the output tokens.

Config: batch_size=32, lr=3e-4, epochs=20, max_seq_len=512
Optimizer: AdamW with cosine LR schedule + 5% warmup
Checkpoints saved every 500 steps, final model to model/weights.npz
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from tokenizers import Tokenizer

# Ensure model/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model.model import AppleScriptTransformer, ModelConfig, count_parameters

# ===========================================================================
# Config
# ===========================================================================
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 20
MAX_SEQ_LEN = 256
WARMUP_FRACTION = 0.05  # warmup for first 5% of total steps
LOG_EVERY = 25
CHECKPOINT_EVERY = 2000
TRAIN_SPLIT = 0.95
SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
WEIGHTS_PATH = MODEL_DIR / "weights.npz"


# ===========================================================================
# Data loading & tokenization
# ===========================================================================

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


def tokenize_pairs(
    records: list[dict],
    tokenizer: Tokenizer,
    max_len: int,
    input_token_id: int,
    output_token_id: int,
    end_token_id: int,
    pad_token_id: int,
) -> tuple[mx.array, mx.array]:
    """
    Tokenize all records into padded sequences and build loss masks.

    Each sequence: <|input|> {input_text} <|output|> {output_text} <|end|>
    Padded to max_len with <|pad|>.

    Returns:
        token_ids: (N, max_len) int32 array
        loss_mask: (N, max_len) float32 array, 1.0 for output positions, 0.0 elsewhere
    """
    all_ids = []
    all_masks = []

    for rec in records:
        input_text = rec.get("input", "")
        output_text = rec.get("output", "")

        # Encode the input and output portions separately so we know their lengths
        input_enc = tokenizer.encode(f" {input_text} ")
        output_enc = tokenizer.encode(f" {output_text} ")

        # Build the full sequence:
        # [<|input|>] [input_tokens...] [<|output|>] [output_tokens...] [<|end|>]
        seq = (
            [input_token_id]
            + input_enc.ids
            + [output_token_id]
            + output_enc.ids
            + [end_token_id]
        )

        # Truncate if too long
        if len(seq) > max_len:
            seq = seq[:max_len]
            # Make sure it ends with <|end|>
            seq[-1] = end_token_id

        # Build loss mask: 1 for output tokens and <|end|>, 0 for input portion
        # The output portion starts after <|output|> token
        input_prefix_len = 1 + len(input_enc.ids) + 1  # <|input|> + input_tokens + <|output|>
        mask = [0.0] * min(input_prefix_len, len(seq))
        if len(seq) > input_prefix_len:
            mask += [1.0] * (len(seq) - input_prefix_len)

        # Pad to max_len
        pad_len = max_len - len(seq)
        seq = seq + [pad_token_id] * pad_len
        mask = mask + [0.0] * pad_len

        all_ids.append(seq)
        all_masks.append(mask)

    token_ids = mx.array(all_ids, dtype=mx.int32)
    loss_mask = mx.array(all_masks, dtype=mx.float32)

    return token_ids, loss_mask


def create_batches(
    token_ids: mx.array,
    loss_mask: mx.array,
    batch_size: int,
    shuffle: bool = True,
):
    """Yield batches of (token_ids, loss_mask)."""
    n = token_ids.shape[0]
    if shuffle:
        perm = mx.random.permutation(n)
        token_ids = token_ids[perm]
        loss_mask = loss_mask[perm]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield token_ids[start:end], loss_mask[start:end]


# ===========================================================================
# Loss function
# ===========================================================================

def loss_fn(model, tokens, mask):
    """
    Compute masked cross-entropy loss.

    tokens: (B, L) input token IDs
    mask:   (B, L) loss mask (1.0 on output positions)

    The model predicts the next token, so:
    - inputs  = tokens[:, :-1]
    - targets = tokens[:, 1:]
    - mask is shifted accordingly: mask[:, 1:]
    """
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    shifted_mask = mask[:, 1:]  # align mask with targets

    logits = model(inputs)  # (B, L-1, vocab_size)

    # Per-token cross-entropy, no reduction
    ce = nn.losses.cross_entropy(logits, targets, reduction="none")  # (B, L-1)

    # Apply mask: only count loss on output tokens
    masked_loss = ce * shifted_mask

    # Mean over non-zero mask positions
    num_tokens = shifted_mask.sum()
    # Avoid division by zero
    loss = masked_loss.sum() / mx.maximum(num_tokens, mx.array(1.0))
    return loss


# ===========================================================================
# Training
# ===========================================================================

def main():
    mx.random.seed(SEED)

    # ---- Load tokenizer ----
    if not TOKENIZER_PATH.exists():
        print(f"Error: tokenizer not found at {TOKENIZER_PATH}", file=sys.stderr)
        print("Run scripts/train_tokenizer.py first.", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with vocab_size={vocab_size}")

    # Get special token IDs
    input_token_id = tokenizer.token_to_id("<|input|>")
    output_token_id = tokenizer.token_to_id("<|output|>")
    end_token_id = tokenizer.token_to_id("<|end|>")
    pad_token_id = tokenizer.token_to_id("<|pad|>")
    print(f"Special tokens: input={input_token_id}, output={output_token_id}, "
          f"end={end_token_id}, pad={pad_token_id}")

    # ---- Load data ----
    seed_records = load_jsonl(DATA_DIR / "seed_pairs.jsonl")
    expanded_records = load_jsonl(DATA_DIR / "expanded_pairs.jsonl")
    all_records = seed_records + expanded_records
    print(f"Loaded {len(seed_records)} seed + {len(expanded_records)} expanded = "
          f"{len(all_records)} total pairs")

    if not all_records:
        print("Error: No training data found.", file=sys.stderr)
        sys.exit(1)

    # ---- Tokenize ----
    print("Tokenizing...")
    token_ids, loss_mask = tokenize_pairs(
        all_records, tokenizer, MAX_SEQ_LEN,
        input_token_id, output_token_id, end_token_id, pad_token_id,
    )
    print(f"Tokenized shape: {token_ids.shape}")

    # ---- Train/val split ----
    n = token_ids.shape[0]
    perm = mx.random.permutation(n)
    token_ids = token_ids[perm]
    loss_mask = loss_mask[perm]

    split = int(n * TRAIN_SPLIT)
    train_ids, val_ids = token_ids[:split], token_ids[split:]
    train_mask, val_mask = loss_mask[:split], loss_mask[split:]
    print(f"Train: {train_ids.shape[0]}, Val: {val_ids.shape[0]}")

    # ---- Initialize model ----
    config = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=MAX_SEQ_LEN,
        pad_token_id=pad_token_id,
        input_token_id=input_token_id,
        output_token_id=output_token_id,
        end_token_id=end_token_id,
    )
    model = AppleScriptTransformer(config)
    mx.eval(model.parameters())

    nparams = count_parameters(model)
    print(f"Model parameters: {nparams / 1e6:.2f}M")

    # ---- Resume from checkpoint if available ----
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Find latest checkpoint
    ckpt_files = sorted(CHECKPOINT_DIR.glob("step_*.npz"))
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        print(f"Resuming from checkpoint: {latest_ckpt}")
        model.load_weights(str(latest_ckpt))
        mx.eval(model.parameters())
        # Extract step number from filename
        step_num = int(latest_ckpt.stem.split("_")[1])
        global_step = step_num
        steps_per_epoch = train_ids.shape[0] // BATCH_SIZE
        start_epoch = global_step // steps_per_epoch
        print(f"  Resumed at step {global_step}, starting from epoch {start_epoch + 1}")

    # ---- Optimizer with warmup + cosine decay ----
    total_steps = (train_ids.shape[0] // BATCH_SIZE) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_FRACTION)
    decay_steps = total_steps - warmup_steps

    print(f"Total steps: {total_steps}, warmup: {warmup_steps}, decay: {decay_steps}")

    # Build LR schedule: linear warmup then cosine decay
    warmup_schedule = optim.linear_schedule(0.0, LEARNING_RATE, steps=warmup_steps)
    cosine_schedule = optim.cosine_decay(LEARNING_RATE, decay_steps, end=LEARNING_RATE * 0.01)
    lr_schedule = optim.join_schedules(
        [warmup_schedule, cosine_schedule],
        [warmup_steps],
    )

    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)

    # Fast-forward the LR schedule to the current step
    if global_step > 0:
        for _ in range(global_step):
            lr_schedule(mx.array(0))  # advance schedule counter
        print(f"  LR schedule advanced to step {global_step}")

    # ---- Training step (eager mode) ----
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def step(tokens, mask):
        loss, grads = loss_and_grad_fn(model, tokens, mask)
        optimizer.update(model, grads)
        return loss

    # ---- Training loop ----
    train_losses = []

    print(f"\nStarting training for {EPOCHS} epochs (from epoch {start_epoch + 1})...")
    print(f"  batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, max_seq_len={MAX_SEQ_LEN}")
    print(f"  log_every={LOG_EVERY}, checkpoint_every={CHECKPOINT_EVERY}")
    print()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.perf_counter()
        epoch_losses = []

        for batch_ids, batch_mask in create_batches(
            train_ids, train_mask, BATCH_SIZE, shuffle=True
        ):
            loss = step(batch_ids, batch_mask)
            mx.eval(loss)

            loss_val = loss.item()
            train_losses.append(loss_val)
            epoch_losses.append(loss_val)
            global_step += 1

            # ---- Log ----
            if global_step % LOG_EVERY == 0:
                avg_loss = sum(train_losses[-LOG_EVERY:]) / len(train_losses[-LOG_EVERY:])
                lr = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, 'item') else optimizer.learning_rate
                print(
                    f"  step {global_step:>6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e}"
                )

            # ---- Checkpoint ----
            if global_step % CHECKPOINT_EVERY == 0:
                ckpt_path = CHECKPOINT_DIR / f"step_{global_step:06d}.npz"
                model.save_weights(str(ckpt_path))
                print(f"  >> Checkpoint saved: {ckpt_path}")

        # ---- End of epoch: validation ----
        epoch_time = time.perf_counter() - epoch_start
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # Validation loss
        val_losses = []
        for batch_ids, batch_mask in create_batches(
            val_ids, val_mask, BATCH_SIZE, shuffle=False
        ):
            vl = loss_fn(model, batch_ids, batch_mask)
            mx.eval(vl)
            val_losses.append(vl.item())
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} | "
            f"train_loss {avg_epoch_loss:.4f} | "
            f"val_loss {avg_val_loss:.4f} | "
            f"time {epoch_time:.1f}s"
        )

        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = CHECKPOINT_DIR / "best.npz"
            model.save_weights(str(best_path))
            print(f"  >> New best model saved: {best_path} (val_loss={best_val_loss:.4f})")

        print()

    # ---- Save final model + config ----
    model.save_weights(str(WEIGHTS_PATH))
    import dataclasses
    config_path = MODEL_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    print(f"Final model saved to {WEIGHTS_PATH}")
    print(f"Config saved to {config_path}")
    print(f"Best val_loss: {best_val_loss:.4f}")
    print(f"Total training steps: {global_step}")
    print("Done.")


if __name__ == "__main__":
    main()
