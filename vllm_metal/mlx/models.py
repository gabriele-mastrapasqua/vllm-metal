# SPDX-License-Identifier: Apache-2.0
"""Pure MLX transformer implementation.

This module provides a complete transformer implementation in MLX,
avoiding any PyTorch dependencies during forward pass.
"""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # RMS norm: x * rsqrt(mean(x^2) + eps) * weight
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        positions = mx.arange(max_seq_len).astype(mx.float32)
        freqs = mx.outer(positions, inv_freq)

        # Cache cos and sin
        self._cos = mx.cos(freqs)
        self._sin = mx.sin(freqs)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Apply rotary embedding to input tensor.

        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
            offset: Position offset for KV cache

        Returns:
            Tensor with rotary embedding applied
        """
        seq_len = x.shape[1]
        cos = self._cos[offset : offset + seq_len]
        sin = self._sin[offset : offset + seq_len]

        # Reshape for broadcasting: [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim/2]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        # Split into pairs and rotate
        x1, x2 = mx.split(x, 2, axis=-1)
        rotated = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

        return rotated


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 4096,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qk_norm = qk_norm

        # Number of query heads per KV head (for GQA)
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # QK normalization (used by Qwen3 and others)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, rms_norm_eps)

        # RoPE
        self.rotary = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            mask: Optional attention mask
            cache: Optional KV cache (key, value) for incremental decoding

        Returns:
            Output tensor and updated cache
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        offset = 0 if cache is None else cache[0].shape[1]
        q = self.rotary(q, offset)
        k = self.rotary(k, offset)

        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Expand KV for GQA: [batch, kv_seq, kv_heads, head_dim] -> [batch, kv_seq, num_heads, head_dim]
        if self.num_queries_per_kv > 1:
            k = mx.repeat(k, self.num_queries_per_kv, axis=2)
            v = mx.repeat(v, self.num_queries_per_kv, axis=2)

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # Use MLX's fast SDPA when available
        try:
            output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
        except AttributeError:
            # Fallback to manual implementation
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            if mask is not None:
                scores = scores + mask
            weights = mx.softmax(scores, axis=-1)
            output = weights @ v

        # Transpose back: [batch, seq_len, num_heads, head_dim]
        output = output.transpose(0, 2, 1, 3)

        # Reshape and project: [batch, seq_len, hidden_size]
        output = output.reshape(batch, seq_len, -1)
        output = self.o_proj(output)

        return output, new_cache


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_seq_len: int,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.attention = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
            rms_norm_eps=rms_norm_eps,
        )
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.attention_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array] | None]:
        # Pre-norm attention
        residual = x
        x = self.attention_norm(x)
        x, new_cache = self.attention(x, mask, cache)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_cache


class MLXTransformer(nn.Module):
    """Pure MLX transformer model for text generation.

    This model runs entirely in MLX with no PyTorch dependencies
    during the forward pass.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_position_embeddings,
                qk_norm=getattr(config, "qk_norm", False),
            )
            for _ in range(config.num_hidden_layers)
        ]

        # Final norm and output
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.output.weight = self.embed.weight

    def __call__(
        self,
        input_ids: mx.array,
        cache: list | None = None,
    ) -> tuple[mx.array, list | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            cache: Optional list of KV caches per layer

        Returns:
            Logits [batch, seq_len, vocab_size] and updated cache
        """
        # Embed tokens
        x = self.embed(input_ids)

        # Create causal mask
        seq_len = input_ids.shape[1]
        if cache is not None and cache[0] is not None:
            # Incremental decoding - no mask needed for single token
            mask = None
        else:
            # Full prefill - create causal mask
            if seq_len > 1:
                mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)
                mask = mask[None, None, :, :]  # [1, 1, seq, seq]
            else:
                mask = None

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Forward through layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, cache[i])
            new_cache.append(layer_cache)

        # Final norm and output projection
        x = self.norm(x)
        logits = self.output(x)

        return logits, new_cache

    def generate_step(
        self,
        input_ids: mx.array,
        cache: list | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[mx.array, list | None]:
        """Generate a single token.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            cache: KV cache from previous steps
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold

        Returns:
            Next token ID and updated cache
        """
        # Forward pass
        logits, cache = self(input_ids, cache)

        # Get logits for last position
        logits = logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

            # Top-p sampling
            if top_p < 1.0:
                sorted_indices = mx.argsort(-logits, axis=-1)
                sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
                probs = mx.softmax(sorted_logits, axis=-1)
                cumsum = mx.cumsum(probs, axis=-1)

                # Find cutoff
                cutoff_mask = cumsum > top_p
                # Keep at least one token
                cutoff_mask = mx.concatenate(
                    [
                        mx.zeros((logits.shape[0], 1), dtype=mx.bool_),
                        cutoff_mask[:, :-1],
                    ],
                    axis=-1,
                )
                sorted_logits = mx.where(cutoff_mask, float("-inf"), sorted_logits)

                # Unsort
                logits = mx.zeros_like(logits)
                logits = mx.put_along_axis(
                    logits, sorted_indices, sorted_logits, axis=-1
                )

            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
        else:
            # Greedy
            next_token = mx.argmax(logits, axis=-1)

        return next_token, cache

    def generate(
        self,
        input_ids: mx.array,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> mx.array:
        """Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            eos_token_id: Optional EOS token to stop generation

        Returns:
            Generated token IDs [batch, total_seq_len]
        """
        # Prefill
        cache = None
        logits, cache = self(input_ids, cache)

        # Get first token
        generated = [input_ids]
        last_logits = logits[:, -1:, :]

        for _ in range(max_tokens):
            # Sample next token
            if temperature > 0:
                probs = mx.softmax(last_logits[:, 0, :] / temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))
            else:
                next_token = mx.argmax(last_logits[:, 0, :], axis=-1)

            next_token = next_token[:, None]  # [batch, 1]
            generated.append(next_token)

            # Check for EOS
            if eos_token_id is not None:
                if mx.all(next_token == eos_token_id):
                    break

            # Forward for next token
            last_logits, cache = self(next_token, cache)

        return mx.concatenate(generated, axis=1)
