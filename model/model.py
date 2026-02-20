"""
Decoder-only transformer for natural language -> AppleScript generation.

Architecture: GPT-style with RoPE positional encoding.
Format: <|input|> natural language command <|output|> applescript code <|end|>

~15-30M parameters with default config:
  6 layers, 384 embedding dim, 6 attention heads, vocab 8192, max seq 512
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|input|>": 1,
    "<|output|>": 2,
    "<|end|>": 3,
}


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    d_ff: int = 1536          # 4 * d_model
    max_seq_len: int = 512
    rope_base: float = 10000.0
    rope_traditional: bool = False
    norm_eps: float = 1e-5
    dropout: float = 0.0      # set >0 for training
    pad_token_id: int = 0
    input_token_id: int = 1
    output_token_id: int = 2
    end_token_id: int = 3

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and KV cache support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=config.rope_traditional,
            base=config.rope_base,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # (B, L, n_heads, head_dim) -> (B, n_heads, L, head_dim)
        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE with cache offset
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot-product attention
        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values)


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network: out = W2(SiLU(W1(x)) * W3(x))"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer decoder block with RMSNorm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.ffn = SwiGLUFFN(config)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # Pre-norm attention with residual
        h = self.norm1(x)
        attn_out, cache = self.attention(h, mask=mask, cache=cache)
        x = x + attn_out

        # Pre-norm FFN with residual
        h = self.norm2(x)
        x = x + self.ffn(h)

        return x, cache


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class AppleScriptTransformer(nn.Module):
    """
    Decoder-only transformer for natural language -> AppleScript generation.

    Training format:
        <|input|> natural language command <|output|> applescript code <|end|>

    The model predicts next tokens autoregressively. During training, compute
    cross-entropy loss on the full sequence (or only on tokens after <|output|>).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass for training.

        Args:
            x: Token ids, shape (B, L).
            mask: Optional pre-computed causal mask. If None, one is created.

        Returns:
            Logits of shape (B, L, vocab_size).
        """
        B, L = x.shape

        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(self.tok_embeddings.weight.dtype)

        h = self.tok_embeddings(x)
        for layer in self.layers:
            h, _ = layer(h, mask=mask)
        h = self.norm(h)

        return self.lm_head(h)

    # -------------------------------------------------------------------
    # Inference utilities
    # -------------------------------------------------------------------

    def _prefill(
        self,
        tokens: mx.array,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Process the prompt and return logits + KV caches."""
        B, L = tokens.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        mask = mask.astype(self.tok_embeddings.weight.dtype)

        h = self.tok_embeddings(tokens)
        cache = []
        for layer in self.layers:
            h, c = layer(h, mask=mask)
            cache.append(c)
        h = self.norm(h)
        logits = self.lm_head(h[:, -1])
        return logits, cache

    def _decode_step(
        self,
        token: mx.array,
        cache: List[Tuple[mx.array, mx.array]],
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Decode a single token using the KV cache."""
        h = self.tok_embeddings(token)
        new_cache = []
        for i, layer in enumerate(self.layers):
            h, c = layer(h, mask=None, cache=cache[i])
            new_cache.append(c)
        h = self.norm(h)
        logits = self.lm_head(h[:, -1])
        return logits, new_cache

    @staticmethod
    def _sample(
        logits: mx.array,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> mx.array:
        """
        Sample a token from logits with temperature and top-p (nucleus) sampling.

        Args:
            logits: Raw logits of shape (B, vocab_size) or (vocab_size,).
            temperature: Sampling temperature. 0 = greedy.
            top_p: Nucleus sampling threshold. 1.0 = no filtering.

        Returns:
            Sampled token ids of shape (B,) or scalar.
        """
        if temperature == 0.0:
            return mx.argmax(logits, axis=-1)

        # Apply temperature
        logits = logits / temperature

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            probs = mx.softmax(logits, axis=-1)
            sorted_indices = mx.argsort(-logits, axis=-1)  # descending
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # Create mask: keep tokens where cumulative prob <= top_p
            # Always keep at least the top token
            sorted_mask = cumulative_probs - sorted_probs >= top_p

            # Map mask back to original positions
            inverse_indices = mx.argsort(sorted_indices, axis=-1)
            token_mask = mx.take_along_axis(sorted_mask, inverse_indices, axis=-1)

            logits = mx.where(token_mask, mx.array(float("-inf")), logits)

        return mx.random.categorical(logits)

    def generate(
        self,
        prompt_tokens: mx.array,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        end_token_id: Optional[int] = None,
    ):
        """
        Generate tokens autoregressively.

        Args:
            prompt_tokens: Input token ids, shape (1, L) or (B, L).
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            end_token_id: Stop generation when this token is produced.
                          Defaults to config.end_token_id.

        Yields:
            mx.array: Each generated token id, shape (B,).
        """
        if end_token_id is None:
            end_token_id = self.config.end_token_id

        # Ensure 2D input
        if prompt_tokens.ndim == 1:
            prompt_tokens = prompt_tokens[None, :]

        # Prefill: process entire prompt
        logits, cache = self._prefill(prompt_tokens)
        token = self._sample(logits, temperature=temperature, top_p=top_p)
        yield token

        # Autoregressive decoding
        for _ in range(max_tokens - 1):
            # Check for end token (batch dim)
            if (token == end_token_id).all().item():
                return

            logits, cache = self._decode_step(token[:, None], cache)
            token = self._sample(logits, temperature=temperature, top_p=top_p)
            yield token

    def generate_text(
        self,
        prompt_tokens: mx.array,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> mx.array:
        """
        Non-streaming generation. Returns all generated token ids as a single array.

        Args:
            prompt_tokens: Input token ids, shape (1, L) or (B, L).
            max_tokens: Maximum number of new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            mx.array of generated token ids, shape (B, num_generated).
        """
        tokens = []
        for tok in self.generate(
            prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(tok)
        if not tokens:
            return mx.array([[]], dtype=mx.int32)
        return mx.stack(tokens, axis=-1)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    model: AppleScriptTransformer,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: Optional[mx.array] = None,
) -> mx.array:
    """
    Compute cross-entropy loss for language modelling.

    Args:
        model: The transformer model.
        inputs: Token ids of shape (B, L) — the input sequence.
        targets: Token ids of shape (B, L) — shifted by 1 from inputs.
        loss_mask: Optional boolean mask of shape (B, L).
                   If provided, loss is only computed where mask is True.
                   Typically True for tokens after <|output|>.

    Returns:
        Scalar loss value.
    """
    logits = model(inputs)
    B, L, V = logits.shape

    # Cross-entropy: -log softmax[target]
    logits = logits.astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, targets)

    if loss_mask is not None:
        loss = loss * loss_mask
        return loss.sum() / (loss_mask.sum() + 1e-8)
    else:
        return loss.mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    nparams = sum(
        x.size for k, x in nn.utils.tree_flatten(model.parameters())
    )
    return nparams


def create_model(config: Optional[ModelConfig] = None) -> AppleScriptTransformer:
    """Create a model with the given (or default) configuration."""
    if config is None:
        config = ModelConfig()
    model = AppleScriptTransformer(config)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# Main — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = ModelConfig()
    model = create_model(config)

    n_params = count_parameters(model)
    print(f"Model config:")
    print(f"  layers:        {config.n_layers}")
    print(f"  d_model:       {config.d_model}")
    print(f"  n_heads:       {config.n_heads}")
    print(f"  head_dim:      {config.head_dim}")
    print(f"  d_ff:          {config.d_ff}")
    print(f"  vocab_size:    {config.vocab_size}")
    print(f"  max_seq_len:   {config.max_seq_len}")
    print(f"  parameters:    {n_params:,} ({n_params / 1e6:.1f}M)")
    print()

    # Simulate a forward pass
    dummy_input = mx.array([[1, 100, 200, 300, 2, 400, 500, 3]])  # (1, 8)
    logits = model(dummy_input)
    print(f"Forward pass:    input {dummy_input.shape} -> logits {logits.shape}")

    # Simulate generation
    prompt = mx.array([[1, 100, 200, 300, 2]])  # <|input|> ... <|output|>
    print(f"Generating from prompt of length {prompt.shape[1]}...")
    generated = []
    for i, tok in enumerate(model.generate(prompt, max_tokens=20, temperature=0.8)):
        generated.append(tok.item())
        if tok.item() == config.end_token_id:
            break
    print(f"Generated {len(generated)} tokens: {generated}")
