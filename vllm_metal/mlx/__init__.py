# SPDX-License-Identifier: Apache-2.0
"""MLX compute backend for vLLM Metal.

This module provides MLX-based implementations of core LLM operations
that are significantly faster than PyTorch MPS on Apple Silicon.
"""

from vllm_metal.mlx.attention import (
    mlx_paged_attention,
    mlx_scaled_dot_product_attention,
)
from vllm_metal.mlx.cache import (
    mlx_copy_blocks,
    mlx_reshape_and_cache,
    mlx_swap_blocks,
)
from vllm_metal.mlx.device import (
    get_mlx_device,
    mlx_clear_cache,
    mlx_synchronize,
)

# New boundary-bridging components
from vllm_metal.mlx.engine import MLXEngine, MLXEngineManager
from vllm_metal.mlx.model_loader import MLXModelConfig, load_mlx_model
from vllm_metal.mlx.models import Attention, FeedForward, MLXTransformer, RMSNorm
from vllm_metal.mlx.norm import (
    mlx_fused_add_rms_norm,
    mlx_layer_norm,
    mlx_rms_norm,
)
from vllm_metal.mlx.rope import (
    mlx_apply_rope,
    mlx_rotary_embedding,
)
from vllm_metal.mlx.tensor_bridge import (
    TensorBridge,
    to_mlx,
    to_torch,
)

__all__ = [
    # Attention
    "mlx_scaled_dot_product_attention",
    "mlx_paged_attention",
    # Cache
    "mlx_reshape_and_cache",
    "mlx_copy_blocks",
    "mlx_swap_blocks",
    # Normalization
    "mlx_rms_norm",
    "mlx_layer_norm",
    "mlx_fused_add_rms_norm",
    # RoPE
    "mlx_rotary_embedding",
    "mlx_apply_rope",
    # Tensor bridge
    "to_mlx",
    "to_torch",
    "TensorBridge",
    # Device
    "get_mlx_device",
    "mlx_synchronize",
    "mlx_clear_cache",
    # Engine (boundary-bridging)
    "MLXEngine",
    "MLXEngineManager",
    # Models
    "MLXTransformer",
    "RMSNorm",
    "Attention",
    "FeedForward",
    # Model loading
    "load_mlx_model",
    "MLXModelConfig",
]
