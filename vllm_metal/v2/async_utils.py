# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible async utilities.

This module provides Metal/MLX-compatible implementations of async
utilities that replace CUDA stream-based operations.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.outputs import (
        LogprobsTensors,
        ModelRunnerOutput,
    )
    from vllm.v1.sample.sampler import SamplerOutput


class MetalAsyncOutput:
    """Async output handler for Metal backend.

    On Metal, we don't have CUDA streams, so this is a synchronous
    implementation. Apple Silicon's unified memory architecture
    makes this less of a bottleneck compared to discrete GPU systems.

    This class matches the interface of vLLM's AsyncOutput but uses
    synchronous operations instead of CUDA async copies.
    """

    def __init__(
        self,
        model_runner_output: "ModelRunnerOutput",
        sampler_output: "SamplerOutput",
        num_sampled_tokens: torch.Tensor,
        copy_stream: Any = None,  # Ignored on Metal
        copy_event: Any = None,  # Ignored on Metal
    ):
        """Initialize async output.

        Args:
            model_runner_output: The model runner output.
            sampler_output: The sampler output with token IDs.
            num_sampled_tokens: Number of tokens sampled per request.
            copy_stream: Ignored on Metal (no CUDA streams).
            copy_event: Ignored on Metal (no CUDA events).
        """
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens

        # On Metal, we do synchronous copies since there's no CUDA stream
        # First, synchronize MPS if available
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        # Copy tensors to CPU synchronously
        self.sampled_token_ids = sampler_output.sampled_token_ids.to("cpu")

        # Handle logprobs if present
        if sampler_output.logprobs_tensors is not None:
            self.logprobs_tensors: LogprobsTensors | None = (
                sampler_output.logprobs_tensors.to_cpu_nonblocking()
            )
        else:
            self.logprobs_tensors = None

        self.num_sampled_tokens_cpu = num_sampled_tokens.to("cpu")

        # Handle prompt logprobs dict
        self.prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        if self.model_runner_output.prompt_logprobs_dict:
            for k, v in self.model_runner_output.prompt_logprobs_dict.items():
                if v is not None:
                    self.prompt_logprobs_dict[k] = v.to_cpu_nonblocking()
                else:
                    self.prompt_logprobs_dict[k] = None

    def get_output(self) -> "ModelRunnerOutput":
        """Get the ModelRunnerOutput for this async output.

        On Metal, this is essentially a no-op since we already
        did the copies synchronously in __init__.

        Returns:
            The ModelRunnerOutput with sampled tokens.
        """
        # Synchronize MPS one more time to be safe
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        num_sampled_tokens_np = self.num_sampled_tokens_cpu.numpy()

        # Convert sampled_token_ids to list format expected by vLLM
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_reqs = len(sampled_token_ids)
        for i in range(num_reqs):
            # Truncate to actual number of sampled tokens
            del sampled_token_ids[i][num_sampled_tokens_np[i] :]

        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()

        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        return self.model_runner_output


@contextmanager
def metal_async_barrier(event: Any = None):
    """Metal-compatible async barrier.

    On Metal, this is essentially a no-op since we use synchronous ops.

    Args:
        event: Ignored on Metal.
    """
    # Synchronize MPS before yielding
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    try:
        yield
    finally:
        # Synchronize again after the block
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


class MetalOutputBuffer:
    """Output buffer for Metal backend.

    This manages output tensors for batch inference, providing
    a similar interface to CUDA-based pinned memory buffers but
    using unified memory instead.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.long,
        device: torch.device | str = "cpu",
    ):
        """Initialize output buffer.

        Args:
            max_batch_size: Maximum batch size.
            max_seq_len: Maximum sequence length.
            dtype: Data type for the buffer.
            device: Device for the buffer (use CPU for unified memory).
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device

        # Pre-allocate buffer
        self.buffer = torch.zeros(
            (max_batch_size, max_seq_len),
            dtype=dtype,
            device=self.device,
        )

        # Track usage
        self.current_batch_size = 0
        self.current_seq_len = 0

    def reset(self) -> None:
        """Reset the buffer for a new batch."""
        self.current_batch_size = 0
        self.current_seq_len = 0

    def write(
        self,
        data: torch.Tensor,
        batch_idx: int,
        seq_idx: int,
    ) -> None:
        """Write data to the buffer.

        Args:
            data: Data to write [seq_len] or scalar.
            batch_idx: Batch index.
            seq_idx: Sequence position.
        """
        if data.dim() == 0:
            # Scalar
            self.buffer[batch_idx, seq_idx] = data
        else:
            # Sequence
            seq_len = data.shape[0]
            self.buffer[batch_idx, seq_idx : seq_idx + seq_len] = data

        self.current_batch_size = max(self.current_batch_size, batch_idx + 1)
        self.current_seq_len = max(self.current_seq_len, seq_idx + data.numel())

    def read(self) -> torch.Tensor:
        """Read the current buffer contents.

        Returns:
            Buffer slice with current data.
        """
        return self.buffer[: self.current_batch_size, : self.current_seq_len]


def metal_async_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
) -> MetalAsyncOutput:
    """Perform async tensor copy on Metal.

    On Apple Silicon with unified memory, this is effectively
    a synchronous operation since CPU and GPU share memory.

    Args:
        src: Source tensor.
        dst: Destination tensor.

    Returns:
        MetalAsyncOutput for the copy operation.
    """
    # Direct copy (no async needed with unified memory)
    dst.copy_(src)
    # This function returns old-style MetalAsyncOutput, not used in new code path
    return None  # type: ignore
