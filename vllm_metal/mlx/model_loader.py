# SPDX-License-Identifier: Apache-2.0
"""MLX model loader - loads HuggingFace weights directly into MLX.

This module provides efficient model loading that converts weights to MLX
format once at load time, avoiding per-layer PyTorch<->MLX bridging.
"""

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

logger = logging.getLogger(__name__)


def download_model(model_name: str, cache_dir: str | None = None) -> Path:
    """Download model from HuggingFace Hub.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-0.6B")
        cache_dir: Optional cache directory

    Returns:
        Path to downloaded model directory
    """
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
    )
    return Path(model_path)


def load_config(model_path: Path) -> dict[str, Any]:
    """Load model configuration from config.json."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found at {model_path}")

    with open(config_path) as f:
        return json.load(f)


def load_weights_to_mlx(model_path: Path) -> dict[str, mx.array]:
    """Load safetensors weights directly into MLX arrays.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary mapping weight names to MLX arrays
    """
    weights = {}

    # Find all safetensor files
    safetensor_files = list(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found at {model_path}")

    for sf_path in safetensor_files:
        # Use MLX framework which handles bfloat16 natively
        try:
            with safe_open(sf_path, framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        except (TypeError, ValueError):
            # Fallback: try pytorch framework and convert
            with safe_open(sf_path, framework="pt") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # Convert PyTorch tensor to MLX
                    if tensor.dtype == torch.bfloat16:
                        # Handle bfloat16 via int16 view
                        np_array = tensor.view(torch.int16).numpy()
                        weights[key] = mx.array(np_array).view(mx.bfloat16)
                    else:
                        weights[key] = mx.array(tensor.numpy())

    logger.info(f"Loaded {len(weights)} weight tensors to MLX")
    return weights


def convert_weight_names(
    weights: dict[str, mx.array], config: dict
) -> dict[str, mx.array]:
    """Convert HuggingFace weight names to our MLX model format.

    Different model architectures use different naming conventions.
    This function normalizes them.
    """
    _ = config  # Reserved for architecture-specific conversions

    # Map common patterns
    converted = {}
    for name, tensor in weights.items():
        new_name = name

        # Common transformations
        new_name = new_name.replace("model.", "")
        new_name = new_name.replace("self_attn", "attention")
        new_name = new_name.replace("mlp", "ffn")
        new_name = new_name.replace("input_layernorm", "attention_norm")
        new_name = new_name.replace("post_attention_layernorm", "ffn_norm")
        new_name = new_name.replace("lm_head", "output")
        new_name = new_name.replace("embed_tokens", "embed")

        converted[new_name] = tensor

    return converted


class MLXModelConfig:
    """Configuration for MLX transformer model."""

    def __init__(self, config: dict[str, Any]):
        self.vocab_size = config.get("vocab_size", 32000)
        self.hidden_size = config.get("hidden_size", 4096)
        self.intermediate_size = config.get("intermediate_size", 11008)
        self.num_hidden_layers = config.get("num_hidden_layers", 32)
        self.num_attention_heads = config.get("num_attention_heads", 32)
        self.num_key_value_heads = config.get(
            "num_key_value_heads", config.get("num_attention_heads", 32)
        )
        self.head_dim = config.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-6)
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.max_position_embeddings = config.get("max_position_embeddings", 4096)
        self.tie_word_embeddings = config.get("tie_word_embeddings", False)

        # QK normalization (used by Qwen3 and others)
        self.qk_norm = config.get("qk_norm", False)

        # Architecture type
        self.model_type = config.get("model_type", "llama")

    def __repr__(self):
        return (
            f"MLXModelConfig(hidden={self.hidden_size}, layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}, kv_heads={self.num_key_value_heads})"
        )


def detect_qk_norm(weights: dict[str, mx.array]) -> bool:
    """Detect if model uses QK normalization from weight names."""
    return any("q_norm" in k or "k_norm" in k for k in weights.keys())


def get_model_class(config: MLXModelConfig):
    """Get the appropriate MLX model class for the architecture."""
    from vllm_metal.mlx.models import MLXTransformer

    return MLXTransformer


def load_mlx_model(
    model_name: str,
    cache_dir: str | None = None,
    dtype: mx.Dtype = mx.float16,
) -> tuple[Any, MLXModelConfig]:
    """Load a HuggingFace model directly into MLX.

    Args:
        model_name: HuggingFace model name
        cache_dir: Optional cache directory
        dtype: Target dtype for weights

    Returns:
        Tuple of (model, config)
    """
    logger.info(f"Loading model {model_name} to MLX...")

    # Download model
    model_path = download_model(model_name, cache_dir)

    # Load config
    hf_config = load_config(model_path)

    # Load weights to MLX
    weights = load_weights_to_mlx(model_path)

    # Detect qk_norm from weights (some models don't have it in config)
    has_qk_norm = detect_qk_norm(weights)
    if has_qk_norm and not hf_config.get("qk_norm", False):
        logger.info("Detected QK normalization from weights")
        hf_config["qk_norm"] = True

    config = MLXModelConfig(hf_config)
    logger.info(f"Model config: {config}, qk_norm={config.qk_norm}")

    # Convert weight names
    weights = convert_weight_names(weights, hf_config)

    # Convert dtype if needed
    if dtype != mx.float32:
        weights = {k: v.astype(dtype) for k, v in weights.items()}

    # Create model
    model_cls = get_model_class(config)
    model = model_cls(config)

    # Load weights into model
    model.load_weights(list(weights.items()))

    # Ensure weights are on GPU
    mx.eval(model.parameters())

    logger.info(f"Model loaded successfully to MLX ({dtype})")
    return model, config
