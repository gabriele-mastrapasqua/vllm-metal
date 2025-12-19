# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible input batch preparation functions.

These functions replace Triton kernels with PyTorch implementations
for the Metal backend, matching the vLLM V1 API signatures.
"""

import torch


def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Prepare inputs for prefill phase using PyTorch operations.

    This replaces the Triton kernel _prepare_prefill_inputs_kernel.
    Copies tokens from prefill_token_ids to input_ids based on the mapping.

    Args:
        input_ids: Output tensor for token IDs [num_tokens]
        next_prefill_tokens: Tensor to store next prefill positions [num_reqs]
        idx_mapping: Maps batch index to request state index [num_reqs]
        query_start_loc: Start location of each query in input_ids [num_reqs + 1]
        prefill_token_ids: Source token IDs [num_reqs, max_prefill_len]
        prefill_len: Prefill length for each request [num_reqs]
        num_computed_tokens: Number of already computed tokens [num_reqs]
    """
    num_reqs = idx_mapping.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()
        prefill_length = prefill_len[req_state_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()

        # Skip if not in prefill phase
        if num_computed >= prefill_length:
            continue

        query_start = query_start_loc[batch_idx].item()
        query_end = query_start_loc[batch_idx + 1].item()
        query_len = query_end - query_start

        if query_len == 0:
            continue

        # Copy tokens from prefill_token_ids to input_ids
        src_start = num_computed
        src_end = num_computed + query_len
        dst_start = query_start
        dst_end = query_start + query_len

        input_ids[dst_start:dst_end] = prefill_token_ids[
            req_state_idx, src_start:src_end
        ]

        # Update next_prefill_tokens
        next_prefill_tokens[req_state_idx] = num_computed + query_len


def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Prepare position and sequence length tensors using PyTorch.

    This replaces the Triton kernel _prepare_pos_seq_lens_kernel.

    Args:
        idx_mapping: Maps batch index to request state index [num_reqs]
        query_start_loc: Start location of each query [num_reqs + 1]
        num_computed_tokens: Number of already computed tokens [num_reqs]
        pos: Output tensor for positions [num_tokens]
        seq_lens: Output tensor for sequence lengths [max_num_reqs]
    """
    num_reqs = idx_mapping.shape[0]
    max_num_reqs = seq_lens.shape[0]

    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()
        num_computed = num_computed_tokens[req_state_idx].item()

        start = query_start_loc[batch_idx].item()
        end = query_start_loc[batch_idx + 1].item()
        query_len = end - start

        # seq_len = num_computed_tokens + query_len
        seq_len = num_computed + query_len
        seq_lens[batch_idx] = seq_len

        # Compute positions starting from num_computed_tokens
        if query_len > 0:
            positions = torch.arange(
                num_computed,
                num_computed + query_len,
                dtype=pos.dtype,
                device=pos.device,
            )
            pos[start:end] = positions

    # Pad unused seq_lens as 0 for full CUDA graphs
    if num_reqs < max_num_reqs:
        seq_lens[num_reqs:max_num_reqs] = 0


def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor:
    """Combine sampled and draft tokens using PyTorch.

    This replaces the Triton kernel _combine_sampled_and_draft_tokens_kernel.

    Args:
        input_ids: Tensor for token IDs [num_tokens]
        idx_mapping: Maps batch index to request state index [num_reqs]
        last_sampled_tokens: Last sampled tokens per request [max_num_reqs]
        query_start_loc: Start location of each query [num_reqs + 1]
        seq_lens: Sequence lengths [num_reqs]
        prefill_len: Prefill length for each request [max_num_reqs]
        draft_tokens: Draft token IDs [max_num_reqs, num_speculative_steps]
        cu_num_logits: Cumulative number of logits [num_reqs + 1]
        num_logits: Total number of logits

    Returns:
        logits_indices: Tensor of logits indices [num_logits]
    """
    num_reqs = seq_lens.shape[0]

    logits_indices = torch.empty(
        num_logits,
        dtype=torch.int64,
        device=input_ids.device,
    )

    for batch_idx in range(num_reqs):
        req_state_idx = idx_mapping[batch_idx].item()

        # Get the number of logits and draft tokens
        cu_num_logits_start = cu_num_logits[batch_idx].item()
        cu_num_logits_end = cu_num_logits[batch_idx + 1].item()
        num_logits_for_req = cu_num_logits_end - cu_num_logits_start
        num_draft_tokens = num_logits_for_req - 1

        # Compute the logits indices
        query_end = query_start_loc[batch_idx + 1].item()
        logits_start = query_end - num_logits_for_req

        for i in range(num_logits_for_req):
            logits_indices[cu_num_logits_start + i] = logits_start + i

        seq_len = seq_lens[batch_idx].item()
        prefill_length = prefill_len[req_state_idx].item()

        if seq_len <= prefill_length:
            # Handling prefill tokens. No sampled or draft tokens.
            continue

        # Write the last sampled token ID to input_ids
        last_token_id = last_sampled_tokens[req_state_idx].item()
        input_ids[query_end - num_logits_for_req] = last_token_id

        # Write the draft tokens (if any) to input_ids
        if num_draft_tokens > 0:
            for i in range(num_draft_tokens):
                draft_token = draft_tokens[req_state_idx, i].item()
                input_ids[query_end - num_draft_tokens + i] = draft_token

    return logits_indices


def post_update(
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_computed_tokens: torch.Tensor,
    # [max_num_reqs]
    last_sampled_tokens: torch.Tensor,
    # [max_num_reqs, vocab_size]
    output_bin_counts: torch.Tensor,
    # [num_reqs, num_speculative_steps + 1]
    sampled_tokens: torch.Tensor,
    # [num_reqs]
    num_sampled: torch.Tensor,
    # [num_reqs]
    num_rejected: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> None:
    """Post-update batch state using PyTorch.

    This replaces the Triton kernel _post_update_kernel.

    Args:
        idx_mapping: Maps batch index to request state index [num_reqs]
        num_computed_tokens: Number of computed tokens per request [max_num_reqs]
        last_sampled_tokens: Last sampled tokens per request [max_num_reqs]
        output_bin_counts: Token bin counts for penalties [max_num_reqs, vocab_size]
        sampled_tokens: Sampled tokens [num_reqs, num_speculative_steps + 1]
        num_sampled: Number of sampled tokens per request [num_reqs]
        num_rejected: Number of rejected tokens per request [num_reqs]
        query_start_loc: Start location of each query [num_reqs + 1]
    """
    num_reqs = idx_mapping.shape[0]

    for req_id in range(num_reqs):
        req_state_idx = idx_mapping[req_id].item()

        n_sampled = num_sampled[req_id].item()
        if n_sampled > 0:
            # Store the last sampled token
            token_id = sampled_tokens[req_id, n_sampled - 1].item()
            last_sampled_tokens[req_state_idx] = token_id

        # Update output_bin_counts for each sampled token
        for i in range(int(n_sampled)):
            token_id = sampled_tokens[req_id, i].item()
            output_bin_counts[req_state_idx, token_id] += 1

        # Update num_computed_tokens
        query_start = query_start_loc[req_id].item()
        query_end = query_start_loc[req_id + 1].item()
        query_len = query_end - query_start
        n_rejected = num_rejected[req_id].item()

        num_computed = num_computed_tokens[req_state_idx].item()
        num_computed += query_len - n_rejected
        num_computed_tokens[req_state_idx] = num_computed
