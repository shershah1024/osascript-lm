"""
Inference script for the osascript-lm model.
Takes natural language input, generates AppleScript, optionally executes it.
"""

import argparse
import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from model.model import AppleScriptTransformer, ModelConfig, count_parameters


def load_model(model_dir: str, checkpoint: str = None):
    """Load the trained model and tokenizer."""
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "config.json")
    weights_path = checkpoint or os.path.join(model_dir, "weights.npz")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load or create config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        # Infer from tokenizer
        vocab_size = tokenizer.get_vocab_size()
        config = ModelConfig(
            vocab_size=vocab_size,
            pad_token_id=tokenizer.token_to_id("<|pad|>"),
            input_token_id=tokenizer.token_to_id("<|input|>"),
            output_token_id=tokenizer.token_to_id("<|output|>"),
            end_token_id=tokenizer.token_to_id("<|end|>"),
        )

    model = AppleScriptTransformer(config)
    weights = mx.load(weights_path)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    return model, tokenizer, config


def generate(
    model,
    tokenizer,
    config,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate AppleScript from a natural language prompt."""
    input_text = f"<|input|> {prompt} <|output|>"
    token_ids = tokenizer.encode(input_text).ids
    prompt_tokens = mx.array([token_ids])

    generated = list(token_ids)
    for tok in model.generate(
        prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        t = tok.item()
        if t == config.end_token_id or t == config.pad_token_id:
            break
        generated.append(t)

    # Decode only the output part
    output_token_id = config.output_token_id
    try:
        output_start = generated.index(output_token_id) + 1
    except ValueError:
        output_start = len(token_ids)

    output_ids = generated[output_start:]
    output_text = tokenizer.decode(output_ids).strip()

    return output_text


def execute_applescript(script: str) -> tuple[bool, str]:
    """Execute an AppleScript and return (success, output)."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Script execution timed out (30s)"
    except Exception as e:
        return False, str(e)


def interactive_mode(model, tokenizer, config, auto_execute: bool = False):
    """Interactive REPL for generating and executing AppleScript."""
    print("osascript-lm — Natural Language → AppleScript")
    print("Type 'quit' to exit, 'help' for commands\n")

    while True:
        try:
            prompt = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "help":
            print("  Type a natural language command to generate AppleScript")
            print("  'quit' — exit")
            print("  'help' — show this message")
            continue

        script = generate(model, tokenizer, config, prompt)

        print(f"\n--- AppleScript ---")
        print(script)
        print("---")

        if auto_execute:
            print("Auto-executing...")
            success, output = execute_applescript(script)
            if success:
                print(f"OK: {output}" if output else "OK")
            else:
                print(f"Error: {output}")
        else:
            try:
                choice = input("Execute? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue

            if choice == "y":
                success, output = execute_applescript(script)
                if success:
                    print(f"OK: {output}" if output else "OK")
                else:
                    print(f"Error: {output}")

        print()


def main():
    parser = argparse.ArgumentParser(description="osascript-lm inference")
    parser.add_argument("--model-dir", default="model", help="Path to model directory")
    parser.add_argument("--checkpoint", default=None, help="Path to specific checkpoint")
    parser.add_argument("--command", type=str, default=None, help="Single command to run")
    parser.add_argument("--execute", action="store_true", help="Execute generated script")
    parser.add_argument("--auto", action="store_true", help="Auto-execute without confirmation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer, config = load_model(args.model_dir, args.checkpoint)
    n = count_parameters(model)
    print(f"Model loaded ({config.n_layers}L, {config.d_model}D, {n/1e6:.1f}M params)\n")

    if args.command:
        script = generate(model, tokenizer, config, args.command,
                         temperature=args.temperature, top_p=args.top_p)
        print(f"Input: {args.command}")
        print(f"AppleScript:\n{script}")

        if args.execute:
            success, output = execute_applescript(script)
            if success:
                print(f"Result: {output}" if output else "Executed successfully")
            else:
                print(f"Error: {output}")
    else:
        interactive_mode(model, tokenizer, config, auto_execute=args.auto)


if __name__ == "__main__":
    main()
