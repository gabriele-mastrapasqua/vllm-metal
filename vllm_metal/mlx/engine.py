# SPDX-License-Identifier: Apache-2.0
"""MLX inference engine with boundary-only bridging.

This module provides the core inference engine that runs entirely in MLX,
with PyTorch bridging only at input/output boundaries.
"""

import logging

import mlx.core as mx
import numpy as np
import torch

from vllm_metal.mlx.model_loader import load_mlx_model

logger = logging.getLogger(__name__)


class MLXEngine:
    """MLX inference engine with boundary-only PyTorch bridging.

    This engine:
    1. Loads model weights directly into MLX (once at init)
    2. Runs entire forward pass in MLX (no per-layer bridging)
    3. Bridges only at input (token IDs) and output (logits)

    Performance characteristics:
    - Model weights: Stored in MLX, never converted
    - Forward pass: 100% MLX operations
    - Input conversion: int64 numpy copy (fast, small data)
    - Output conversion: float32 numpy copy (only final logits)
    """

    def __init__(
        self,
        model_name: str,
        dtype: mx.Dtype = mx.float16,
        max_batch_size: int = 8,
        max_seq_len: int = 4096,
    ):
        """Initialize the MLX engine.

        Args:
            model_name: HuggingFace model name
            dtype: Model dtype (float16 recommended)
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
        """
        self.model_name = model_name
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # Load model directly to MLX
        self.model, self.config = load_mlx_model(model_name, dtype=dtype)

        # KV cache storage (per request)
        self._caches: dict[int, list] = {}

        logger.info(f"MLXEngine initialized: {model_name}")
        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Layers: {self.config.num_hidden_layers}")
        logger.info(f"  Heads: {self.config.num_attention_heads}")
        logger.info(f"  KV Heads: {self.config.num_key_value_heads}")

    def _torch_to_mlx_ids(self, token_ids: torch.Tensor) -> mx.array:
        """Convert PyTorch token IDs to MLX array.

        This is a lightweight conversion - just copying int64 values.
        For a batch of 8 sequences of 512 tokens, this is only 32KB.
        """
        # Ensure on CPU and convert via numpy
        if token_ids.device.type != "cpu":
            token_ids = token_ids.cpu()
        return mx.array(token_ids.numpy())

    def _mlx_to_torch_logits(
        self,
        logits: mx.array,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Convert MLX logits to PyTorch tensor.

        Only converts the final logits, not intermediate activations.
        For vocab_size=32000 and batch=8, this is ~1MB.
        """
        mx.eval(logits)
        np_logits = np.array(logits)
        tensor = torch.from_numpy(np_logits)
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Run forward pass with boundary-only bridging.

        Args:
            input_ids: Token IDs [batch, seq_len] (PyTorch)
            cache_ids: Optional cache IDs for each request in batch

        Returns:
            Tuple of (logits, cache_ids) where logits is PyTorch tensor
        """
        batch_size = input_ids.shape[0]

        # === BRIDGE IN: PyTorch -> MLX (token IDs only) ===
        mlx_input_ids = self._torch_to_mlx_ids(input_ids)

        # Get or create caches
        if cache_ids is None:
            cache_ids = list(range(batch_size))
            caches = [None] * batch_size
        else:
            caches = [self._caches.get(cid) for cid in cache_ids]

        # === MLX FORWARD PASS (no bridging) ===
        # For simplicity, process batch together if all have same cache state
        # In production, you'd want proper batched cache handling
        if batch_size == 1:
            logits, new_cache = self.model(mlx_input_ids, caches[0])
            new_caches = [new_cache]
        else:
            # Process each request separately (simple but not optimal)
            all_logits = []
            new_caches = []
            for i in range(batch_size):
                single_input = mlx_input_ids[i : i + 1]
                single_logits, single_cache = self.model(single_input, caches[i])
                all_logits.append(single_logits)
                new_caches.append(single_cache)
            logits = mx.concatenate(all_logits, axis=0)

        # Store updated caches
        for cid, cache in zip(cache_ids, new_caches, strict=False):
            self._caches[cid] = cache

        # === BRIDGE OUT: MLX -> PyTorch (logits only) ===
        torch_logits = self._mlx_to_torch_logits(logits)

        return torch_logits, cache_ids

    def prefill(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        """Prefill phase - process full prompt.

        Args:
            input_ids: Prompt token IDs [batch, seq_len]

        Returns:
            Tuple of (logits, cache_ids)
        """
        return self.forward(input_ids, cache_ids=None)

    def decode(
        self,
        input_ids: torch.Tensor,
        cache_ids: list[int],
    ) -> torch.Tensor:
        """Decode phase - process single token with cache.

        Args:
            input_ids: Single token IDs [batch, 1]
            cache_ids: Cache IDs from prefill

        Returns:
            Logits tensor
        """
        logits, _ = self.forward(input_ids, cache_ids)
        return logits

    def clear_cache(self, cache_id: int | None = None):
        """Clear KV cache.

        Args:
            cache_id: Specific cache to clear, or None for all
        """
        if cache_id is None:
            self._caches.clear()
        elif cache_id in self._caches:
            del self._caches[cache_id]

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens with boundary-only bridging.

        This is a simple generation loop for testing.
        In production, vLLM's scheduler handles this.

        Args:
            input_ids: Prompt token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            eos_token_id: EOS token ID

        Returns:
            Generated token IDs [batch, total_len]
        """
        # === BRIDGE IN (once) ===
        mlx_input_ids = self._torch_to_mlx_ids(input_ids)

        # === ENTIRE GENERATION IN MLX ===
        output_ids = self.model.generate(
            mlx_input_ids,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # === BRIDGE OUT (once) ===
        mx.eval(output_ids)
        np_output = np.array(output_ids)
        torch_output = torch.from_numpy(np_output)

        return torch_output


class MLXEngineManager:
    """Manager for MLX engines, handling model loading and caching."""

    _engines: dict[str, MLXEngine] = {}

    @classmethod
    def get_engine(
        cls,
        model_name: str,
        dtype: mx.Dtype = mx.float16,
    ) -> MLXEngine:
        """Get or create an MLX engine for a model.

        Args:
            model_name: HuggingFace model name
            dtype: Model dtype

        Returns:
            MLXEngine instance
        """
        key = f"{model_name}:{dtype}"
        if key not in cls._engines:
            cls._engines[key] = MLXEngine(model_name, dtype=dtype)
        return cls._engines[key]

    @classmethod
    def clear_engines(cls):
        """Clear all cached engines."""
        cls._engines.clear()
