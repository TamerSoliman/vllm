# ANNOTATED: vLLM BlockTable - PagedAttention Memory Mapping
# Source: vllm/v1/worker/block_table.py
#
# ============================================================================
# WHAT: BlockTable translates logical token positions → physical memory blocks
# HOW: Maintains a 2D mapping (request_id, token_position) → physical_block_id
# WHY: Enables PagedAttention: operating-system-style paging for KV cache
# ============================================================================

"""
PAGED ATTENTION: The Memory Innovation

Traditional KV Cache (Contiguous Memory):
- Allocate max_seq_len slots per request upfront
- Problem: Massive waste for short completions
- Example: Allocate 2048 slots, use 50 → 97.5% wasted!

PagedAttention (Virtual Memory Paging):
- Divide KV cache into fixed-size blocks (e.g., 16 tokens each)
- Allocate blocks on-demand as tokens are generated
- Non-contiguous physical blocks, contiguous logical view
- ANALOGY: Exactly like OS virtual memory paging!

MEMORY SAVINGS:
- Traditional: max_seq_len × num_requests × KV_size
- PagedAttention: actual_tokens × KV_size / block_size (rounded up)
- Typical savings: 3-4x reduction in memory waste

EXAMPLE:
Request with 50 tokens (block_size=16):
- Traditional: Allocate 2048 slots (wasteful)
- PagedAttention: Allocate 4 blocks (4 × 16 = 64 slots, only 28% waste)

Logical Token Positions:  [0, 1, 2, ..., 15] [16, 17, ..., 31] [32, ..., 47] [48, 49]
Logical Block IDs:        [    Block 0    ] [    Block 1    ] [  Block 2  ] [Block 3]
Physical Block IDs:       [      42       ] [      17       ] [     88    ] [   3   ]
Physical Memory Offset:   [   42 × 16    ] [   17 × 16    ] [  88 × 16  ] [ 3 × 16]

Notice: Non-contiguous physical blocks (42, 17, 88, 3) but logically contiguous!
"""

import numpy as np
import torch

from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


class BlockTable:
    """
    ========================================================================
    BLOCK TABLE: Logical-to-Physical Mapping for PagedAttention
    ========================================================================

    WHAT: Maps (request_id, token_position) → physical_memory_slot

    HOW:
    1. Maintain 2D table: block_table[request_idx][block_idx] = physical_block_id
    2. Compute slot_mapping[token_idx] = physical_block_id × block_size + offset

    WHY: PagedAttention kernel needs physical memory addresses

    ANALOGY: Operating System Page Table
    - Logical address (request, token_pos) → Virtual Page
    - block_table maps virtual pages → Physical frames
    - slot_mapping gives actual memory address

    CRITICAL OPERATIONS:
    - append_row: Add blocks to request's block table
    - compute_slot_mapping: Convert token positions → memory addresses
    - commit_*: Copy CPU tables to GPU for attention kernel
    """

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        dcp_kv_cache_interleave_size: int,
    ):
        """
        INITIALIZATION: Create block table buffers

        KEY PARAMETERS:
        - block_size: How many tokens per memory block (e.g., 16)
          ANALOGY: OS page size (e.g., 4KB pages)

        - max_num_reqs: Maximum concurrent requests
          ANALOGY: Maximum number of processes in OS

        - max_num_blocks_per_req: Maximum blocks per request
          ANALOGY: Maximum virtual pages per process

        - max_num_batched_tokens: Maximum tokens in a single batch
          WHY: Determines slot_mapping buffer size

        HYBRID BLOCKS:
        - block_size: Memory allocation granularity
        - kernel_block_size: Attention kernel granularity
        - If different: Memory blocks are subdivided for kernel
        - Example: 32-token memory blocks, 16-token kernel blocks
          → Each memory block = 2 kernel blocks
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        # ================================================================
        # HANDLE HYBRID BLOCK SIZES
        # ================================================================
        # WHAT: Memory block size may differ from kernel block size
        # WHY: Some kernels require specific block sizes
        # HOW: Map memory blocks → multiple kernel blocks

        if kernel_block_size == block_size:
            # Standard case: Direct 1:1 mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: Subdivision required
            # Example: block_size=32, kernel_block_size=16
            # → 1 memory block = 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size  # Use kernel size for all calculations
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        # ================================================================
        # ALLOCATE BLOCK TABLE
        # ================================================================
        # WHAT: 2D table mapping (request, block_index) → physical_block_id
        # SHAPE: [max_num_reqs, max_num_blocks_per_req]
        # ANALOGY: Page table in OS (virtual page → physical frame)

        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        # Track how many blocks each request has
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # ================================================================
        # ALLOCATE SLOT MAPPING
        # ================================================================
        # WHAT: 1D array mapping token_index → physical_memory_slot
        # SHAPE: [max_num_batched_tokens]
        # WHY: Attention kernel needs physical addresses for each token

        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        # Precompute offsets for hybrid block mapping
        if self.use_hybrid_blocks:
            # Example: blocks_per_kv_block=2 → [0, 1]
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        # Decode Context Parallelism (DCP) support
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.dcp_kv_cache_interleave_size = dcp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        """
        ====================================================================
        APPEND BLOCKS TO REQUEST'S BLOCK TABLE
        ====================================================================

        WHAT: Add newly allocated blocks to request's page table
        HOW: Copy block_ids to block_table[row_idx, start:end]
        WHY: When request generates more tokens, need more blocks

        ARGUMENTS:
        - block_ids: List of physical block IDs (from BlockPool)
        - row_idx: Request index in block table

        EXAMPLE:
        Request has 2 blocks: [42, 17]
        append_row([88, 3], row_idx=5)
        → block_table[5, :] = [42, 17, 88, 3, ...]

        ANALOGY: Allocating more virtual pages to a process
        """
        if not block_ids:
            return

        # ================================================================
        # HYBRID BLOCKS: Map memory blocks → kernel blocks
        # ================================================================
        # WHAT: If using hybrid blocks, subdivide each memory block
        # HOW: map_to_kernel_blocks([0, 1]) → [0, 1, 2, 3] (if 2 per block)
        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        # Append to block table
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """
        WHAT: Initialize request's block table (replace existing)
        WHY: For new requests or when resetting block table
        """
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        """
        ====================================================================
        COMPUTE PHYSICAL MEMORY ADDRESSES FOR TOKENS
        ====================================================================

        WHAT: Convert (request_id, token_position) → physical_memory_slot
        HOW:
        1. token_position ÷ block_size = block_index
        2. block_table[request_idx][block_index] = physical_block_id
        3. physical_slot = physical_block_id × block_size + offset

        WHY: PagedAttention kernel needs to know WHERE to read/write KV cache

        ARGUMENTS:
        - req_indices: Which request each token belongs to (shape: [num_tokens])
        - positions: Token position within each request (shape: [num_tokens])

        OUTPUT:
        - slot_mapping: Physical memory slot for each token (shape: [num_tokens])

        EXAMPLE:
        req_indices = [0, 0, 1, 1, 1]
        positions = [0, 1, 0, 1, 2]
        block_size = 2

        Step 1: Compute block indices
        block_indices = positions // block_size = [0, 0, 0, 0, 1]

        Step 2: Lookup physical block IDs
        block_table[0, :] = [42, 17, ...]
        block_table[1, :] = [88, 3, ...]
        → physical_blocks = [42, 42, 88, 88, 3]

        Step 3: Compute offsets
        offsets = positions % block_size = [0, 1, 0, 1, 0]

        Step 4: Final slot mapping
        slot_mapping = physical_blocks × block_size + offsets
                     = [42×2+0, 42×2+1, 88×2+0, 88×2+1, 3×2+0]
                     = [84, 85, 176, 177, 6]

        OPERATING SYSTEM ANALOGY:
        This is exactly virtual-to-physical address translation!
        - Virtual address = (page_number, offset)
        - Physical address = page_table[page_number] × page_size + offset
        """

        if self.dcp_world_size > 1:
            # ============================================================
            # DECODE CONTEXT PARALLELISM (DCP) MAPPING
            # ============================================================
            # WHAT: Distribute KV cache across multiple GPUs
            # HOW: Interleaved storage (round-robin across ranks)
            # WHY: Parallel decode for long context

            virtual_block_size = self.block_size * self.dcp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]

            # Determine which tokens belong to this rank
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.dcp_kv_cache_interleave_size
                % self.dcp_world_size
                == self.dcp_rank
            )

            # Calculate local block offsets
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.dcp_kv_cache_interleave_size)
                * self.dcp_kv_cache_interleave_size
                + virtual_block_offsets % self.dcp_kv_cache_interleave_size
            )

            # Compute slot mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Mark non-local tokens with -1
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            # ============================================================
            # STANDARD (NON-DCP) MAPPING
            # ============================================================
            # This is the core PagedAttention address calculation

            # STEP 1: Calculate which block each token belongs to
            # block_table_indices = request_idx × max_blocks + block_idx
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            )

            # STEP 2: Lookup physical block IDs from block table
            # Ravel flattens 2D table for efficient indexing
            block_numbers = self.block_table.np.ravel()[block_table_indices]

            # STEP 3: Calculate offset within block
            block_offsets = positions % self.block_size

            # STEP 4: Compute final physical slot
            # physical_slot = physical_block_id × block_size + offset
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )

    def commit_block_table(self, num_reqs: int) -> None:
        """
        WHAT: Copy block table from CPU to GPU
        WHY: Attention kernel runs on GPU, needs GPU-side table
        HOW: Async memcpy (non-blocking)
        """
        self.block_table.copy_to_gpu(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        """
        WHAT: Copy slot mapping from CPU to GPU
        WHY: Attention kernel needs physical addresses on GPU
        HOW: Async memcpy (non-blocking)
        """
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        """Reset block table (for benchmarking or cleanup)"""
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """
        ====================================================================
        HYBRID BLOCKS: Map Memory Blocks → Kernel Blocks
        ====================================================================

        WHAT: Convert memory block IDs to kernel block IDs
        WHY: Memory allocation uses larger blocks than attention kernel

        EXAMPLE:
        Memory block size: 32 tokens
        Kernel block size: 16 tokens
        → blocks_per_kv_block = 2

        Input: kv_manager_block_ids = [0, 1, 2]
        Output: kernel_block_ids = [0, 1, 2, 3, 4, 5]

        MAPPING:
        Memory block 0 → Kernel blocks [0, 1]
        Memory block 1 → Kernel blocks [2, 3]
        Memory block 2 → Kernel blocks [4, 5]

        HOW:
        kernel_block_id = memory_block_id × blocks_per_kv_block + offset
        where offset ∈ [0, blocks_per_kv_block)
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        # Reshape to add offset dimension
        # [0, 1, 2] → [[0], [1], [2]]
        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange  # [[0, 1]]
        )
        # Result: [[0, 1], [2, 3], [4, 5]]

        # Flatten back to 1D
        return kernel_block_ids.reshape(-1)
        # [0, 1, 2, 3, 4, 5]

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Get GPU-side block table (for attention kernel)"""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Get CPU-side block table"""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Get NumPy view of block table (for CPU manipulation)"""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        """
        WHAT: Create dual CPU/GPU buffer for block table or slot mapping
        WHY: Build table on CPU, then copy to GPU for kernel
        HOW: Pinned memory for fast async transfers
        """
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# ============================================================================
# KEY TAKEAWAYS: BlockTable & PagedAttention
# ============================================================================
#
# 1. VIRTUAL MEMORY PAGING FOR KV CACHE
#    - PagedAttention applies OS paging concepts to LLM serving
#    - Logical token positions mapped to non-contiguous physical blocks
#    - Eliminates memory fragmentation and waste
#
# 2. MEMORY SAVINGS
#    - Traditional: Pre-allocate max_seq_len slots per request
#    - PagedAttention: Allocate blocks on-demand as tokens generated
#    - Typical savings: 3-4x reduction in wasted memory
#
# 3. BLOCK TABLE = PAGE TABLE
#    - 2D mapping: [request_idx][block_idx] → physical_block_id
#    - Exactly analogous to OS page table
#    - Maintained on CPU, copied to GPU for attention kernel
#
# 4. SLOT MAPPING = ADDRESS TRANSLATION
#    - Convert (request, token_pos) → physical_memory_slot
#    - Formula: physical_block_id × block_size + offset
#    - This is the attention kernel's memory access pattern
#
# 5. DUAL CPU/GPU BUFFERS
#    - Block table built on CPU (flexible, easy to update)
#    - Copied to GPU before each forward pass (fast async transfer)
#    - Pinned memory for maximum transfer bandwidth
#
# 6. HYBRID BLOCKS
#    - Memory allocation granularity may differ from kernel granularity
#    - Example: 32-token memory blocks, 16-token kernel blocks
#    - map_to_kernel_blocks() handles subdivision
#
# 7. DECODE CONTEXT PARALLELISM (DCP)
#    - Distribute KV cache across multiple GPUs
#    - Interleaved storage for load balancing
#    - Each rank processes subset of tokens
#
# 8. PERFORMANCE CRITICAL
#    - compute_slot_mapping() called every iteration
#    - Optimized with NumPy vectorized operations
#    - Must be extremely fast to avoid becoming bottleneck
#
# ============================================================================

"""
VISUAL EXAMPLE: PagedAttention in Action

Request A: "Explain quantum computing" (3 blocks allocated)
Request B: "Hello world" (1 block allocated)

LOGICAL VIEW (What user/model sees):
Request A: Tokens [0, 1, ..., 15] [16, 17, ..., 31] [32, 33, ..., 47]
Request B: Tokens [0, 1, ..., 15]

PHYSICAL MEMORY (Actual GPU VRAM layout):
Block 0 (physical): Request C data (not currently running)
Block 1 (physical): Request B tokens [0-15]
Block 2 (physical): Free
Block 3 (physical): Request A tokens [32-47]
Block 4 (physical): Free
...
Block 17 (physical): Request A tokens [16-31]
...
Block 42 (physical): Request A tokens [0-15]

BLOCK TABLE:
Request A (row 0): [42, 17, 3]
Request B (row 5): [1]

SLOT MAPPING CALCULATION (for Request A, token 20):
1. block_idx = 20 // 16 = 1
2. physical_block_id = block_table[0][1] = 17
3. offset = 20 % 16 = 4
4. physical_slot = 17 × 16 + 4 = 276

Attention kernel reads KV cache at physical_slot 276!

BENEFIT: Request A's tokens non-contiguous in memory (blocks 42, 17, 3)
         but logically contiguous to model → No fragmentation!
"""
