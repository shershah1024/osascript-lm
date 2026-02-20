# osascript-lm

A 20.5M parameter decoder-only transformer that converts natural language commands into AppleScript, built from scratch and trained on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

The model runs entirely on-device with sub-second inference on M1 Macs. It handles system controls, timers, app management, file operations, music playback, and system info queries — while correctly routing out-of-scope requests (messaging, email, calendar, tasks, knowledge questions) to a cloud LLM via a `PASS_TO_CLOUD` output.

## Architecture

| Parameter | Value |
|---|---|
| Type | Decoder-only transformer (GPT-style) |
| Parameters | 20,452,224 (20.5M) |
| Layers | 6 |
| Embedding dim | 384 |
| Attention heads | 6 |
| FFN dim | 1536 (SwiGLU) |
| Vocab size | 8192 (ByteLevel BPE) |
| Max sequence length | 256 tokens |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Normalization | RMSNorm |
| Model size | 78 MB (weights.npz) |

### Components

- **CausalSelfAttention** — Multi-head attention with RoPE and KV caching for efficient autoregressive generation
- **SwiGLU FFN** — Gated feed-forward network (`SwiGLU(x) = (xW₁ ⊙ σ(xW_gate)) W₂`)
- **RMSNorm** — Pre-norm architecture (norm before attention and FFN)
- **KV Cache** — Streaming token generation during inference

### Token format

```
<|input|> natural language command <|output|> applescript code <|end|>
```

Special tokens: `<|input|>` (0), `<|output|>` (1), `<|end|>` (2), `<|pad|>` (3)

Training loss is masked to only compute cross-entropy on output tokens (after `<|output|>`).

## Capabilities

### On-device (generates AppleScript)

| Category | Examples |
|---|---|
| **Timers** | "set a timer for 25 minutes" → `do shell script "sleep 1500"` + notification |
| **App control** | "open Chrome", "quit Safari", "switch to Finder" |
| **Volume** | "set volume to 50 percent" → `set volume output volume 50` |
| **Brightness** | "set brightness to 75%" → `do shell script "brightness 0.75"` |
| **Dark mode** | "turn on dark mode" → `set dark mode to true` |
| **WiFi/Bluetooth** | "turn off wifi" → `networksetup -setairportpower en0 off` |
| **Sleep/Lock** | "lock my screen" → `keystroke "q" using {command down, control down}` |
| **Screenshots** | "take a screenshot" → `screencapture` |
| **Music** | "next song", "pause music", "shuffle on" |
| **File operations** | "empty the trash", "open Documents folder" |
| **System info** | "what's my IP", "battery level", "uptime", "disk space" |

### Cloud-routed (outputs `PASS_TO_CLOUD`)

Messaging, email, calendar, reminders, tasks, knowledge questions, creative writing, code help, analysis, math/reasoning.

## Quickstart

### Requirements

- Python 3.12+
- Apple Silicon Mac (M1/M2/M3)
- MLX, tokenizers, httpx

```bash
pip install mlx tokenizers httpx
```

### Single command

```bash
python scripts/inference.py --model-dir model --command "set a timer for 10 minutes"
```

### Interactive REPL

```bash
python scripts/inference.py --model-dir model
```

### Auto-execute mode

```bash
python scripts/inference.py --model-dir model --auto
```

### Python API

```python
from model.model import AppleScriptTransformer, ModelConfig
from scripts.inference import load_model, generate

model, tokenizer, config = load_model("model")
script = generate(model, tokenizer, config, "open Safari")
# → 'tell application "Safari" to activate'
```

## Training

### Data

27,853 training pairs generated using Azure OpenAI (gpt-5-mini) across focused pipelines:

| Pipeline | Pairs | Description |
|---|---|---|
| timers_durations | 2,500 | Timer/countdown commands with numeric extraction |
| calendar_schedule | 2,388 | Calendar events (reclassified → PASS_TO_CLOUD) |
| app_control | 1,500 | Open/close/switch/force-quit apps |
| system_control | 2,000 | Volume, brightness, dark mode, wifi, bluetooth |
| files_finder | 1,500 | File operations, Finder commands |
| communication | 1,515 | Messaging/email (reclassified → PASS_TO_CLOUD) |
| media_music | 1,500 | Music playback controls |
| info_queries | 1,000 | System information queries |
| reminders_notes | 1,500 | Reminders/notes (reclassified → PASS_TO_CLOUD) |
| negative_oos | 2,476 | Out-of-scope → PASS_TO_CLOUD |
| **conv_*** | **9,578** | **Longer conversational variants of all categories** |
| seed | 396 | Hand-crafted seed pairs |

396 seed pairs + 27,457 expanded = 27,853 total. The conversational pipelines specifically target verbose, natural phrasing (8+ words) to improve robustness to filler words and polite requests.

### Training config

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 (peak) |
| LR schedule | Linear warmup + cosine decay |
| Warmup steps | 1,653 |
| Batch size | 16 |
| Epochs | 20 |
| Sequence length | 256 (0% truncation) |
| Framework | MLX (eager mode) |
| Hardware | Apple M1, 8GB RAM |
| Training time | ~5.5 hours |

### Tokenizer

ByteLevel BPE trained on the full dataset using HuggingFace `tokenizers`. Vocabulary of 8,192 tokens with 4 reserved special token IDs.

## Accuracy

Tested across 42+ commands:

| Input type | Accuracy |
|---|---|
| Short commands ("open Safari") | ~92% |
| Long conversational ("hey can you open Chrome for me please") | ~85% |
| Timer math (number extraction) | ~98% |
| PASS_TO_CLOUD routing | ~95% |

### Known limitations

- Vague relative commands ("make it brighter") without specific values can produce inconsistent output
- Very long sentences with lots of filler may occasionally confuse app names
- "Empty trash" sometimes generates "open trash" in verbose phrasing

## Project structure

```
osascript-lm/
├── model/
│   ├── model.py          # Transformer architecture (20.5M params)
│   ├── config.json        # Model hyperparameters
│   ├── tokenizer.json     # BPE tokenizer (8192 vocab)
│   ├── weights.npz        # Trained weights (78 MB)
│   └── __init__.py
├── scripts/
│   ├── inference.py       # Inference + interactive REPL
│   ├── train.py           # Training loop with checkpointing
│   ├── train_tokenizer.py # BPE tokenizer training
│   ├── expand_data_azure.py       # Data generation (10 pipelines)
│   └── expand_conversational.py   # Conversational data generation
├── data/
│   └── seed_pairs.jsonl   # 396 hand-crafted seed pairs
└── README.md
```

## License

Private — not for redistribution.
