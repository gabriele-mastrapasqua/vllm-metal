# SPDX-License-Identifier: Apache-2.0
"""Metal-compatible BlockTables implementation.

This module provides a BlockTables implementation that uses pure PyTorch
operations instead of Triton kernels, enabling vLLM V1 to run on Apple Silicon.
"""

from collections.abc import Iterable

import torch
from vllm.v1.utils import CpuGpuBuffer

# Pad slot ID (same as vLLM)
PAD_SLOT_ID = -1


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


class MetalBlockTables:
    """Metal-compatible BlockTables using pure PyTorch operations.

    This is a drop-in replacement for vLLM's BlockTables class that uses
    PyTorch operations instead of Triton kernels.
    """

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.block_sizes = block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.device = device
        # MPS doesn't support pinned memory
        self.pin_memory = False if str(device).startswith("mps") else pin_memory

        self.num_kv_cache_groups = len(self.block_sizes)

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[torch.Tensor] = []
        for i in range(self.num_kv_cache_groups):
            block_size = self.block_sizes[i]
            max_num_blocks = cdiv(self.max_model_len, block_size)
            block_table = torch.zeros(
                self.max_num_reqs,
                max_num_blocks,
                dtype=torch.int32,
                device=self.device,
            )
            self.block_tables.append(block_table)
        self.block_table_ptrs = self._make_ptr_tensor(self.block_tables)

        # Block tables used for model's forward pass
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(block_table) for block_table in self.block_tables
        ]
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

        self.block_table_strides = torch.tensor(
            [b.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.num_blocks = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        # Misc buffers
        self.req_indices = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.overwrite = self._make_buffer(self.max_num_reqs, dtype=torch.bool)
        self.cu_num_new_blocks = self._make_buffer(
            self.num_kv_cache_groups, self.max_num_reqs + 1, dtype=torch.int32
        )

    def _make_buffer(self, *args, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *args, dtype=dtype, pin_memory=self.pin_memory, device=self.device
        )

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        ptrs_tensor_cpu = torch.tensor(
            [t.data_ptr() for t in x],
            dtype=torch.uint64,
            device="cpu",
            pin_memory=self.pin_memory,
        )
        return ptrs_tensor_cpu.to(self.device, non_blocking=True)

    def append_block_ids(
        self,
        req_indices: list[int],
        cu_num_new_blocks: tuple[list[int], ...],
        new_block_ids: tuple[list[int], ...],
        overwrite: list[bool],
    ) -> None:
        """Append new block IDs to block tables using pure PyTorch."""
        num_reqs = len(req_indices)

        for group_id in range(self.num_kv_cache_groups):
            block_table = self.block_tables[group_id]

            for batch_idx in range(num_reqs):
                req_idx = req_indices[batch_idx]
                do_overwrite = overwrite[batch_idx]

                start_idx = cu_num_new_blocks[group_id][batch_idx]
                end_idx = cu_num_new_blocks[group_id][batch_idx + 1]
                num_new_blocks = end_idx - start_idx

                if num_new_blocks == 0:
                    continue

                # Get current num_blocks for this request
                if do_overwrite:
                    dst_start_idx = 0
                else:
                    dst_start_idx = self.num_blocks[group_id, req_idx].item()

                dst_end_idx = dst_start_idx + num_new_blocks
                self.num_blocks[group_id, req_idx] = dst_end_idx

                # Copy block IDs to block table
                block_ids = new_block_ids[group_id][start_idx:end_idx]
                block_table[req_idx, dst_start_idx:dst_end_idx] = torch.tensor(
                    block_ids, dtype=torch.int32, device=self.device
                )

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Gather block tables for batched requests using pure PyTorch.

        Returns:
            Tuple of input block tables, each sliced to num_reqs rows.
        """
        num_reqs = idx_mapping.shape[0]

        for group_id in range(self.num_kv_cache_groups):
            src_block_table = self.block_tables[group_id]
            dst_block_table = self.input_block_tables[group_id]

            for batch_idx in range(num_reqs):
                req_idx = idx_mapping[batch_idx].item()
                num_blks = self.num_blocks[group_id, req_idx].item()

                # Copy the block table row
                dst_block_table[batch_idx, :num_blks] = src_block_table[
                    req_idx, :num_blks
                ]
                # Zero out the rest
                if num_blks < dst_block_table.shape[1]:
                    dst_block_table[batch_idx, num_blks:] = 0

        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute slot mappings for attention using pure PyTorch.

        Args:
            query_start_loc: Cumulative token counts [num_reqs + 1]
            positions: Token positions [num_tokens]

        Returns:
            Slot mappings tensor [num_kv_cache_groups, num_tokens]
        """
        num_reqs = query_start_loc.shape[0] - 1
        num_tokens = positions.shape[0]

        for group_id in range(self.num_kv_cache_groups):
            block_table = self.input_block_tables[group_id]
            page_size = self.block_sizes[group_id]

            for req_idx in range(num_reqs):
                start_idx = query_start_loc[req_idx].item()
                end_idx = query_start_loc[req_idx + 1].item()

                for i in range(start_idx, end_idx):
                    pos = positions[i].item()
                    block_idx = pos // page_size
                    block_offset = pos % page_size

                    # Get physical block number from input_block_tables (gathered)
                    block_number = block_table[req_idx, block_idx].item()
                    slot_id = block_number * page_size + block_offset
                    self.slot_mappings[group_id, i] = slot_id

            # Pad remaining slots to PAD_SLOT_ID
            if num_tokens < self.max_num_batched_tokens:
                self.slot_mappings[group_id, num_tokens:] = PAD_SLOT_ID

        return self.slot_mappings[:, :num_tokens]

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        """Get dummy block tables for warmup."""
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        """Get dummy slot mappings for warmup."""
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]
