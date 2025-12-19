# SPDX-License-Identifier: Apache-2.0
"""Metal V2 Model Runner - extends GPU model runner for Metal/MLX backend."""

from contextlib import contextmanager

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Module-level patching for Metal (must happen before importing GPUModelRunner)
# ============================================================================


def _patched_bincount_metal(
    prefill_token_ids: torch.Tensor,
    prefill_len: int,
    prompt_len: int,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """PyTorch-based bincount replacement for Metal (no Triton)."""
    prompt_bin_mask.zero_()
    output_bin_counts.zero_()

    # Get the tokens in the range [prompt_len, prefill_len)
    if prefill_len > prompt_len:
        tokens = prefill_token_ids[prompt_len:prefill_len]
        tokens_cpu = tokens.cpu().to(torch.int64)
        vocab_size = output_bin_counts.shape[0]
        counts = torch.bincount(tokens_cpu, minlength=vocab_size)
        min_len = min(len(counts), vocab_size)
        output_bin_counts[:min_len] = counts[:min_len].to(output_bin_counts.device)

    # Set prompt_bin_mask for tokens in [0, prompt_len)
    if prompt_len > 0:
        prompt_tokens = prefill_token_ids[:prompt_len]
        prompt_tokens_cpu = prompt_tokens.cpu().to(torch.int64)
        vocab_size = prompt_bin_mask.shape[0]
        for token in prompt_tokens_cpu:
            if 0 <= token < vocab_size:
                prompt_bin_mask[token] = 1


# Patch bincount BEFORE any vLLM modules that use it are imported
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    penalties_module.bincount = _patched_bincount_metal
    logger.debug("Patched penalties_module.bincount for Metal")
except ImportError:
    pass

# Patch states module for MPS compatibility (UVA / unified memory)
# Apple Silicon has true unified memory, so we can use regular tensors
try:
    import vllm.v1.worker.gpu.states as states_module

    # Apple Silicon has unified memory - similar to UVA
    states_module.is_uva_available = lambda: True

    # Patch UvaBuffer to work with MPS unified memory
    class _MetalUvaBuffer:
        """MPS-compatible UvaBuffer using unified memory.

        Apple Silicon has true unified memory - CPU and GPU share the same
        physical memory. However, PyTorch MPS doesn't share memory between
        CPU and MPS tensors like CUDA UVA does.

        Solution: Use CPU tensors for both cpu and gpu attributes.
        MPS operations can accept CPU tensors directly (with auto data movement).
        This ensures writes to cpu/np are immediately visible to gpu.
        """

        def __init__(self, *size, dtype):
            # Keep everything on CPU - this is the source of truth
            self.cpu = torch.zeros(*size, dtype=dtype, device="cpu")
            self.np = self.cpu.numpy()
            # gpu is the same tensor - MPS can accept CPU tensors
            # This ensures all modifications are visible to both
            self.gpu = self.cpu

    states_module.UvaBuffer = _MetalUvaBuffer
    logger.debug("Patched states_module for Metal unified memory")
except ImportError as e:
    logger.warning(f"Failed to patch states module: {e}")

# =============================================================================
# Patch BlockTables with Metal-compatible implementation (pure PyTorch)
# =============================================================================
try:
    import vllm.v1.worker.gpu.block_table as block_table_module

    from vllm_metal.v2.metal_block_table import MetalBlockTables

    # Replace the entire BlockTables class with our Metal implementation
    block_table_module.BlockTables = MetalBlockTables
    logger.debug("Patched BlockTables with MetalBlockTables for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch block_table module: {e}")

# Patch input_batch functions BEFORE importing GPUModelRunner
try:
    import vllm.v1.worker.gpu.input_batch as input_batch_module

    from vllm_metal.v2.input_batch import (
        combine_sampled_and_draft_tokens,
        post_update,
        prepare_pos_seq_lens,
        prepare_prefill_inputs,
    )

    input_batch_module.prepare_prefill_inputs = prepare_prefill_inputs
    input_batch_module.prepare_pos_seq_lens = prepare_pos_seq_lens
    input_batch_module.combine_sampled_and_draft_tokens = (
        combine_sampled_and_draft_tokens
    )
    input_batch_module.post_update = post_update
    logger.debug("Patched input_batch module functions for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch input_batch module: {e}")

# Patch penalties module - uses Triton kernels
try:
    import vllm.v1.worker.gpu.sample.penalties as penalties_module

    from vllm_metal.v2.penalties import apply_penalties_and_temperature

    penalties_module.apply_penalties_and_temperature = apply_penalties_and_temperature
    logger.debug("Patched penalties_module.apply_penalties_and_temperature for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch penalties module: {e}")

# Patch gumbel module - uses Triton kernel
try:
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_module

    from vllm_metal.v2.gumbel import gumbel_sample

    gumbel_module.gumbel_sample = gumbel_sample
    logger.debug("Patched gumbel_module.gumbel_sample for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch gumbel module: {e}")

# Patch async_utils module - uses CUDA streams
try:
    import vllm.v1.worker.gpu.async_utils as async_utils_module

    from vllm_metal.v2.async_utils import MetalAsyncOutput

    async_utils_module.AsyncOutput = MetalAsyncOutput
    logger.debug("Patched async_utils_module.AsyncOutput for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch async_utils module: {e}")


# =============================================================================
# Mock classes for CUDA compatibility on MPS
# =============================================================================


class _MockCudaStream:
    """Mock CUDA stream for MPS compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, stream):
        pass

    def synchronize(self):
        torch.mps.synchronize()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _MockCudaEvent:
    """Mock CUDA event for MPS compatibility."""

    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def synchronize(self):
        torch.mps.synchronize()

    def query(self):
        return True


class _MockCudaGraphManager:
    """Mock CudaGraphManager for MPS compatibility."""

    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.device = device
        self.pool = None
        self.cudagraph_mode = None
        self.capture_sizes = []
        self.disabled = True

    def capture(self, *args, **kwargs):
        pass

    def replay(self, *args, **kwargs):
        return None

    def get_graph(self, *args, **kwargs):
        return None

    def should_use_cudagraph(self, *args, **kwargs):
        return False

    def get_cudagraph_size(self, *args, **kwargs):
        # Return None to indicate no cudagraph should be used
        return None

    def get_cudagraph(self, *args, **kwargs):
        return None


# Patch CudaGraphManager BEFORE importing GPUModelRunner
try:
    import vllm.v1.worker.gpu.cudagraph_utils as cudagraph_utils_module

    cudagraph_utils_module.CudaGraphManager = _MockCudaGraphManager
    logger.debug("Patched CudaGraphManager for Metal")
except ImportError as e:
    logger.warning(f"Failed to patch cudagraph_utils: {e}")


# Now import the rest of vLLM modules (they will get our patched functions)
from vllm.model_executor.model_loader import get_model  # noqa: E402
from vllm.v1.kv_cache_interface import KVCacheConfig  # noqa: E402
from vllm.v1.utils import CpuGpuBuffer  # noqa: E402
from vllm.v1.worker.gpu.attn_utils import (  # noqa: E402
    init_attn_backend,
    init_kv_cache,
)

# Use our Metal-compatible BlockTables (pure PyTorch, no Triton)
# Note: block_table_module.BlockTables is already patched above
from vllm.v1.worker.gpu.block_table import BlockTables  # noqa: E402
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402

# Patch states module's bincount reference
try:
    import vllm.v1.worker.gpu.states as states_module

    states_module.bincount = _patched_bincount_metal
    logger.debug("Patched states_module.bincount for Metal")
except (ImportError, AttributeError):
    pass


@contextmanager
def _patch_cuda_for_mps():
    """Context manager to patch CUDA stream/event/graph for MPS compatibility.

    vLLM's GPUModelRunner.__init__ creates CUDA streams, events and graphs.
    We temporarily replace them with MPS-compatible mocks.
    """
    original_stream = torch.cuda.Stream
    original_event = torch.cuda.Event
    original_current_stream = torch.cuda.current_stream
    original_graph_pool = getattr(torch.cuda, "graph_pool_handle", None)

    try:
        torch.cuda.Stream = _MockCudaStream  # type: ignore[assignment]
        torch.cuda.Event = _MockCudaEvent  # type: ignore[assignment]
        torch.cuda.current_stream = lambda device=None: _MockCudaStream()  # type: ignore[assignment,return-value]
        torch.cuda.graph_pool_handle = lambda: None  # type: ignore[assignment,return-value]
        yield
    finally:
        torch.cuda.Stream = original_stream
        torch.cuda.Event = original_event
        torch.cuda.current_stream = original_current_stream
        if original_graph_pool is not None:
            torch.cuda.graph_pool_handle = original_graph_pool


class MetalModelRunner(GPUModelRunner):
    """Metal/MLX model runner that extends the GPU model runner.

    This class inherits all the complex input batch management, attention
    metadata building, and model execution from GPUModelRunner. It only
    overrides Metal-specific functionality like:
    - Disabling CUDA-specific features (pinned memory, CUDA graphs)
    - Using MPS/MLX synchronization instead of CUDA
    - Metal-specific device handling
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Patch CUDA stream/event to work with MPS before calling parent init
        with _patch_cuda_for_mps():
            super().__init__(vllm_config, device)

        # Override CUDA-specific settings
        self.pin_memory = False  # Metal uses unified memory
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace the mock streams with our MPS-compatible versions
        self.output_copy_stream = _MockCudaStream()

        # Replace GPU tensors with MPS equivalents
        self._postprocess_tensors()

        # Log initialization
        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.model_config.get_hidden_size()}, "
            f"num_heads={self.model_config.get_num_attention_heads(self.parallel_config)}, "
            f"num_kv_heads={self.model_config.get_num_kv_heads(self.parallel_config)}, "
            f"head_dim={self.model_config.get_head_size()}, "
            f"block_size={self.cache_config.block_size}"
        )

    def _postprocess_tensors(self) -> None:
        """Replace GPU tensors with device tensors for Metal."""
        # For Metal, we don't need separate CPU and GPU buffers
        # since MPS/MLX uses unified memory
        for v in vars(self).values():
            if isinstance(v, CpuGpuBuffer):
                v.gpu = v.cpu

    def _sync_device(self) -> None:
        """Synchronize the MPS/MLX device instead of CUDA."""
        import mlx.core as mx

        mx.eval([])  # Force MLX evaluation
        torch.mps.synchronize()

    def load_model(self, *args, **kwargs) -> None:
        """Load the model to the MPS device."""
        logger.info("Loading model with MLX acceleration...")

        # Load model using standard vLLM loader
        self.model = get_model(
            vllm_config=self.vllm_config,
        )

        # Move model to MPS device
        self.model = self.model.to(self.device)

        logger.info("Model loaded successfully")

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache for Metal backend.

        This overrides GPUModelRunner's method to remove the FLASH_ATTN check.
        Metal backend uses its own attention implementation.
        """
        from copy import deepcopy

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
        )

        # Metal backend - no FLASH_ATTN check needed
        logger.info(
            f"Metal attention backends initialized: {list(self.attn_backends.keys())}"
        )

        self.kv_caches: list[torch.Tensor] = []
        init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore
