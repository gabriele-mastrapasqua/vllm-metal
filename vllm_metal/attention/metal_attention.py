# SPDX-License-Identifier: Apache-2.0
"""Metal attention implementation using MLX."""

from typing import Any

import torch

from vllm_metal._compat import AttentionImpl, init_logger
from vllm_metal.attention.backend import MetalAttentionMetadata

logger = init_logger(__name__)


class MetalAttentionImpl(AttentionImpl):
    """Metal attention implementation using MLX.

    This implementation provides efficient attention computation on
    Apple Silicon by using MLX's optimized SDPA. It handles both
    prefill (variable length) and decode (single token) phases.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: dict[str, Any] | None = None,
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
    ):
        """Initialize Metal attention.

        Args:
            num_heads: Number of attention heads.
            head_size: Size of each head.
            scale: Attention scale factor.
            num_kv_heads: Number of key-value heads (for GQA).
            alibi_slopes: Optional ALiBi slopes.
            sliding_window: Optional sliding window size.
            kv_cache_dtype: Data type for KV cache.
            blocksparse_params: Block sparse parameters (not supported).
            logits_soft_cap: Logits soft cap (not supported).
            attn_type: Attention type (decoder, encoder, etc.).
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        # GQA setup
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Validate unsupported features
        if blocksparse_params is not None:
            logger.warning("Block sparse attention not supported on Metal")
        if logits_soft_cap is not None:
            logger.warning("Logits soft cap not supported on Metal")
        if sliding_window is not None:
            logger.warning("Sliding window attention has limited support on Metal")

        logger.debug(
            f"MetalAttentionImpl initialized: "
            f"num_heads={num_heads}, head_size={head_size}, "
            f"num_kv_heads={num_kv_heads}, scale={scale}"
        )

    def forward(
        self,
        layer: Any,  # AttentionLayer, but we avoid import cycle
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: MetalAttentionMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Args:
            layer: The attention layer (unused on Metal).
            query: Query tensor [num_tokens, num_heads, head_size].
            key: Key tensor [num_tokens, num_kv_heads, head_size].
            value: Value tensor [num_tokens, num_kv_heads, head_size].
            kv_cache: Tuple of (key_cache, value_cache).
            attn_metadata: Attention metadata.
            output: Optional pre-allocated output tensor.

        Returns:
            Attention output [num_tokens, num_heads, head_size].
        """
        key_cache, value_cache = kv_cache

        # Store new K/V into cache
        if attn_metadata.slot_mapping is not None:
            self._store_kv_cache(
                key, value, key_cache, value_cache, attn_metadata.slot_mapping
            )

        # Route to appropriate attention implementation
        if attn_metadata.is_all_prefill:
            attn_output = self._prefill_attention(
                query,
                key,
                value,
                attn_metadata,
            )
        elif attn_metadata.is_all_decode:
            attn_output = self._decode_attention(
                query,
                key_cache,
                value_cache,
                attn_metadata,
            )
        else:
            # Mixed batch - handle prefill and decode separately
            attn_output = self._mixed_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata,
            )

        # Copy to output if provided
        if output is not None:
            output.copy_(attn_output)
            return output

        return attn_output

    def _store_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store key and value tensors into the cache.

        Uses vectorized operations compatible with torch.compile.

        Args:
            key: Key tensor [num_tokens, num_kv_heads, head_size].
            value: Value tensor [num_tokens, num_kv_heads, head_size].
            key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
            value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
            slot_mapping: Slot indices [num_tokens].
        """
        # Reshape cache to [num_blocks * block_size, num_kv_heads, head_size]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
        flat_key_cache = key_cache.view(-1, num_kv_heads, head_size)
        flat_value_cache = value_cache.view(-1, num_kv_heads, head_size)

        # Use vectorized indexing - slot_mapping directly indexes into flat cache
        # This avoids .item() calls and works with torch.compile
        flat_key_cache[slot_mapping] = key
        flat_value_cache[slot_mapping] = value

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for prefill phase using PyTorch SDPA.

        For vLLM V1 with chunked prefill, batches typically contain a single
        sequence. This implementation uses pure PyTorch operations for
        torch.compile compatibility.

        Args:
            query: Query tensor [total_tokens, num_heads, head_size].
            key: Key tensor [total_tokens, num_kv_heads, head_size].
            value: Value tensor [total_tokens, num_kv_heads, head_size].
            attn_metadata: Attention metadata.

        Returns:
            Attention output [total_tokens, num_heads, head_size].
        """
        # For single-sequence batch (common with chunked prefill),
        # treat all tokens as one sequence with causal masking

        # Reshape: [total_tokens, num_heads, head_size] -> [1, num_heads, seq, head_size]
        q = query.transpose(0, 1).unsqueeze(0)
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        # Expand KV for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Use PyTorch SDPA with causal masking
        # Note: is_causal=True creates the proper triangular mask
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale, is_causal=True
        )

        # Reshape back: [1, num_heads, seq, head_size] -> [seq, num_heads, head_size]
        out = out.squeeze(0).transpose(0, 1)

        return out

    def _decode_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for decode phase using PyTorch paged attention.

        This implementation gathers K/V from the paged cache and computes
        attention using pure PyTorch operations for torch.compile compatibility.

        Args:
            query: Query tensor [batch, num_heads, head_size].
            key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
            value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
            attn_metadata: Attention metadata.

        Returns:
            Attention output [batch, num_heads, head_size].
        """
        batch_size = query.shape[0]
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
        block_table = attn_metadata.block_table  # [batch, max_blocks]
        seq_lens = attn_metadata.seq_lens  # [batch]

        # For decode, we have one query token per sequence
        # Gather K/V from cache using block_table
        # IMPORTANT: Use actual max sequence length from seq_lens, not max_model_len
        assert seq_lens is not None, "seq_lens required for decode"
        max_seq_len = int(seq_lens.max().item())

        # Flatten cache for indexing: [total_slots, num_kv_heads, head_size]
        flat_key_cache = key_cache.view(-1, num_kv_heads, head_size)
        flat_value_cache = value_cache.view(-1, num_kv_heads, head_size)

        # Compute slot indices from block_table
        # block_table contains block IDs, we need to convert to slot indices
        assert block_table is not None, "block_table required for decode"
        max_blocks_per_seq = block_table.shape[1]

        # Create position indices within blocks: [max_seq_len]
        positions = torch.arange(max_seq_len, device=query.device)
        block_indices = positions // block_size  # which block for each position
        offsets = positions % block_size  # offset within block

        # Gather block IDs for all sequences and positions
        # block_table: [batch, max_blocks], block_indices: [max_seq_len]
        # We need block_table[b, block_indices[p]] for each batch b and position p
        block_indices_clamped = block_indices.clamp(max=max_blocks_per_seq - 1)
        gathered_blocks = block_table[
            :, block_indices_clamped.long()
        ]  # [batch, max_seq_len]

        # Convert to flat slot indices
        slot_indices = gathered_blocks * block_size + offsets  # [batch, max_seq_len]

        # Gather K/V: [batch, max_seq_len, num_kv_heads, head_size]
        slot_indices_flat = slot_indices.view(-1).long()
        k_gathered = flat_key_cache[slot_indices_flat].view(
            batch_size, max_seq_len, num_kv_heads, head_size
        )
        v_gathered = flat_value_cache[slot_indices_flat].view(
            batch_size, max_seq_len, num_kv_heads, head_size
        )

        # Create attention mask based on sequence lengths
        # Mask out positions beyond seq_len for each batch
        position_ids = torch.arange(max_seq_len, device=query.device).unsqueeze(
            0
        )  # [1, max_seq_len]
        attn_mask = position_ids < seq_lens.unsqueeze(1)  # [batch, max_seq_len]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_seq_len]

        # Prepare tensors for SDPA
        # Query: [batch, num_heads, 1, head_size]
        q = query.unsqueeze(2)  # [batch, num_heads, 1, head_size]

        # K/V: [batch, num_kv_heads, max_seq_len, head_size]
        k = k_gathered.transpose(1, 2)  # [batch, num_kv_heads, max_seq_len, head_size]
        v = v_gathered.transpose(1, 2)

        # Expand KV for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Convert mask to attention bias format (0 for valid, -inf for masked)
        attn_bias = torch.where(
            attn_mask,
            torch.zeros(1, device=query.device, dtype=query.dtype),
            torch.full((1,), float("-inf"), device=query.device, dtype=query.dtype),
        )

        # Compute attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=self.scale
        )

        # Remove sequence dimension: [batch, num_heads, head_size]
        out = out.squeeze(2)

        return out

    def _mixed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Handle mixed prefill/decode batch.

        For torch.compile compatibility, this delegates to _prefill_attention
        which uses pure PyTorch operations.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            key_cache: Key cache (unused, K/V already in query inputs).
            value_cache: Value cache (unused).
            attn_metadata: Attention metadata.

        Returns:
            Attention output.
        """
        # Use the torch.compile compatible prefill attention
        return self._prefill_attention(query, key, value, attn_metadata)
