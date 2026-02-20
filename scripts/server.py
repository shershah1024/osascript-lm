#!/usr/bin/env python3
"""
HTTP server wrapping osascript-lm inference.

Exposes two endpoints:
  GET  /health            → {"status": "ok"}
  POST /generate          → {"script": "...", "is_cloud": false}

Designed to be called from Rune's Swift ToolDispatcher as a drop-in
replacement for llama-server (FunctionGemmaRouter pattern).

The model stays loaded in memory for fast inference (~50-100ms per query).
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from tokenizers import Tokenizer
from model.model import AppleScriptTransformer, ModelConfig, count_parameters


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
MODEL = None
TOKENIZER = None
CONFIG = None


def load_model(model_dir: str):
    global MODEL, TOKENIZER, CONFIG

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "weights.npz")

    TOKENIZER = Tokenizer.from_file(tokenizer_path)

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    CONFIG = ModelConfig(**config_dict)

    MODEL = AppleScriptTransformer(CONFIG)
    weights = mx.load(weights_path)
    MODEL.load_weights(list(weights.items()))
    mx.eval(MODEL.parameters())

    n = count_parameters(MODEL)
    print(f"Model loaded: {CONFIG.n_layers}L, {CONFIG.d_model}D, {n/1e6:.1f}M params")


def generate(prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
    """Generate AppleScript from natural language. Returns the raw output string."""
    input_text = f"<|input|> {prompt} <|output|>"
    token_ids = TOKENIZER.encode(input_text).ids
    prompt_tokens = mx.array([token_ids])

    generated = list(token_ids)
    for tok in MODEL.generate(
        prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    ):
        t = tok.item()
        if t == CONFIG.end_token_id or t == CONFIG.pad_token_id:
            break
        generated.append(t)

    output_token_id = CONFIG.output_token_id
    try:
        output_start = generated.index(output_token_id) + 1
    except ValueError:
        output_start = len(token_ids)

    output_ids = generated[output_start:]
    return TOKENIZER.decode(output_ids).strip()


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class OsascriptHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default access logs; only log errors
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/generate":
            self._send_json({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON"}, 400)
            return

        query = data.get("query", "").strip()
        if not query:
            self._send_json({"error": "missing 'query' field"}, 400)
            return

        temperature = data.get("temperature", 0.0)

        try:
            script = generate(query, temperature=temperature)
            is_cloud = script.strip() == "PASS_TO_CLOUD"

            self._send_json({
                "script": script,
                "is_cloud": is_cloud,
                "query": query,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 500)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="osascript-lm HTTP server")
    parser.add_argument("--model-dir", default="model", help="Path to model directory")
    parser.add_argument("--port", type=int, default=39284, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    load_model(args.model_dir)

    # Warmup
    print("Warming up...")
    _ = generate("hello")

    server = HTTPServer((args.host, args.port), OsascriptHandler)
    print(f"Server ready on http://{args.host}:{args.port}")
    print(f"  GET  /health   → health check")
    print(f"  POST /generate → {{\"query\": \"...\"}}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
