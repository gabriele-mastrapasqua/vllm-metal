# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible penalty and temperature application.

This module provides PyTorch/MLX implementations of penalty functions
that replace Triton kernels on the Metal backend.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import SamplingMetadata


def apply_penalties_and_temperature(
    logits: torch.Tensor,
    sampling_metadata: "SamplingMetadata",
) -> None:
    """Apply penalties and temperature to logits (in-place).

    This function matches vLLM's signature and replaces the Triton kernel
    version with a pure PyTorch implementation for Metal compatibility.

    Args:
        logits: Raw model logits [num_reqs, vocab_size]. Modified in-place.
        sampling_metadata: Sampling metadata containing penalty values.
    """
    num_reqs, vocab_size = logits.shape

    # Extract values from sampling_metadata
    temperatures = sampling_metadata.temperature
    repetition_penalties = sampling_metadata.repetition_penalty
    presence_penalties = sampling_metadata.presence_penalty
    frequency_penalties = sampling_metadata.frequency_penalty

    # Check if we need to apply any penalties (not just temperature)
    needs_penalties = (
        (repetition_penalties != 1.0).any()
        or (presence_penalties != 0.0).any()
        or (frequency_penalties != 0.0).any()
    )

    if needs_penalties:
        # Get the bin data for penalty application
        idx_mapping = sampling_metadata.idx_mapping
        output_bin_counts = (
            sampling_metadata.output_bin_counts
        )  # [max_num_reqs, vocab_size]

        for req_idx in range(num_reqs):
            slot_idx = int(idx_mapping[req_idx])

            # Get penalty values for this request
            rep_penalty = float(repetition_penalties[req_idx])
            pres_penalty = float(presence_penalties[req_idx])
            freq_penalty = float(frequency_penalties[req_idx])

            # Get output counts for this request (tracks generated tokens)
            if output_bin_counts is not None and slot_idx < output_bin_counts.shape[0]:
                output_counts = output_bin_counts[slot_idx]  # [vocab_size]

                # Only use output_counts if it matches vocab_size
                if output_counts.shape[0] == vocab_size:
                    # Apply repetition penalty to generated tokens
                    if rep_penalty != 1.0:
                        token_mask = output_counts > 0
                        if token_mask.any():
                            pos_mask = token_mask & (logits[req_idx] > 0)
                            neg_mask = token_mask & (logits[req_idx] <= 0)
                            logits[req_idx, pos_mask] = (
                                logits[req_idx, pos_mask] / rep_penalty
                            )
                            logits[req_idx, neg_mask] = (
                                logits[req_idx, neg_mask] * rep_penalty
                            )

                    # Apply presence penalty (subtract for each unique token)
                    if pres_penalty != 0.0:
                        token_mask = (output_counts > 0).float()
                        logits[req_idx] = logits[req_idx] - pres_penalty * token_mask

                    # Apply frequency penalty (subtract proportional to count)
                    if freq_penalty != 0.0:
                        logits[req_idx] = (
                            logits[req_idx] - freq_penalty * output_counts.float()
                        )

    # Apply temperature scaling (always needed)
    for req_idx in range(num_reqs):
        temp = float(temperatures[req_idx])
        if temp > 0:
            logits[req_idx] = logits[req_idx] / temp


def apply_temperature(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits [batch_size, vocab_size].
        temperatures: Temperature values [batch_size].

    Returns:
        Temperature-scaled logits.
    """
    # Handle temperature = 0 (greedy) separately
    temp_mask = temperatures > 0
    if temp_mask.any():
        logits[temp_mask] = logits[temp_mask] / temperatures[temp_mask].unsqueeze(-1)

    return logits


def apply_top_k(
    logits: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        top_k: Top-k values [batch_size].

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    batch_size = logits.shape[0]
    vocab_size = logits.shape[1]

    for i in range(batch_size):
        k = int(top_k[i])
        if k > 0 and k < vocab_size:
            # Get threshold value
            top_k_values, _ = torch.topk(logits[i], k)
            threshold = top_k_values[-1]

            # Mask out values below threshold
            logits[i] = torch.where(
                logits[i] >= threshold,
                logits[i],
                torch.full_like(logits[i], float("-inf")),
            )

    return logits


def apply_top_p(
    logits: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        top_p: Top-p values [batch_size].

    Returns:
        Filtered logits with low-probability values set to -inf.
    """
    batch_size = logits.shape[0]

    for i in range(batch_size):
        p = float(top_p[i])
        if p < 1.0:
            # Sort logits and get cumulative probabilities
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Find cutoff index
            sorted_mask = cumulative_probs > p
            # Shift mask to keep at least one token
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            # Apply mask
            sorted_logits[sorted_mask] = float("-inf")

            # Restore original order
            logits[i] = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    return logits
