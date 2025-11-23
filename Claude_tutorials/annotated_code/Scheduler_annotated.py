# ANNOTATED: vLLM Scheduler - Continuous Batching Engine
# Source: vllm/v1/core/sched/scheduler.py
#
# ============================================================================
# WHAT: The Scheduler is the core batching algorithm that decides WHICH
#       requests to process in each forward pass
# HOW: Implements continuous batching with priority scheduling, preemption,
#      and intelligent KV cache management
# WHY: Maximizes GPU utilization and throughput by dynamically batching
#      requests with varying prompt/output lengths
# ============================================================================

"""
CONTINUOUS BATCHING: The Key Innovation

Traditional batching (static batching):
- Wait for N requests
- Process all N prompts together (prefill)
- Generate tokens for all N requests in lockstep
- PROBLEM: Short completions waste GPU waiting for long ones

Continuous batching (vLLM innovation):
- Dynamically add/remove requests from batch at EVERY iteration
- Request A finishing? Immediately replace with new request B
- BENEFIT: Near-100% GPU utilization, 10-20x higher throughput

EXAMPLE:
Iteration 1: [Request A (prompt), Request B (prompt)]
Iteration 2: [Request A (gen 1), Request B (gen 1), Request C (prompt)]
Iteration 3: [Request A (gen 2), Request B DONE, Request C (gen 1), Request D (prompt)]

Notice:
- Prefill (prompt) and decode (generation) in SAME batch
- Requests added/removed dynamically
- Batch composition changes every iteration
"""

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager, compute_encoder_budget
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    """
    ========================================================================
    CONTINUOUS BATCHING SCHEDULER
    ========================================================================

    WHAT: Decides which requests to process in each iteration

    HOW:
    1. Check RUNNING requests first (highest priority)
    2. Try to add WAITING requests if resources available
    3. PREEMPT low-priority requests if out of KV cache memory
    4. Return SchedulerOutput with batch assignments

    WHY: Maximize GPU utilization while respecting:
    - Token budget (max_num_batched_tokens)
    - KV cache capacity (num_gpu_blocks)
    - Request priorities
    - LoRA adapter limits

    CRITICAL INVARIANTS:
    - Never exceed max_num_batched_tokens per iteration
    - Never allocate more KV cache than available
    - Always prioritize RUNNING over WAITING requests
    - Maintain fairness (FCFS within same priority)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        INITIALIZATION: Set up scheduling constraints and managers

        KEY CONFIGURATION:
        - max_num_seqs: Maximum concurrent requests (running queue size)
        - max_num_batched_tokens: Token budget per iteration
        - max_model_len: Maximum sequence length (prompt + output)

        CORE COMPONENTS:
        - KVCacheManager: Manages paged memory blocks for KV cache
        - RequestQueues: WAITING and RUNNING queues
        - EncoderCacheManager: For multimodal/encoder-decoder models
        """
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config

        # ================================================================
        # SCHEDULING CONSTRAINTS
        # ================================================================
        # These are the hard limits the scheduler must respect

        # Maximum number of requests in RUNNING state simultaneously
        # e.g., 256 means at most 256 requests can be actively generating
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs

        # Maximum tokens to process in a single forward pass
        # e.g., 8192 means batch can contain up to 8192 tokens total
        # This prevents OOM and controls latency
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens

        # Maximum sequence length (prompt + generated tokens)
        self.max_model_len = vllm_config.model_config.max_model_len

        # ================================================================
        # REQUEST QUEUES
        # ================================================================
        # WHAT: Organize requests by state (WAITING vs RUNNING)
        # WHY: Different scheduling logic for each state

        # Storage for all requests (req_id -> Request)
        self.requests: dict[str, Request] = {}

        # Scheduling policy (FCFS, PRIORITY, etc.)
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e

        # WAITING queue: New requests not yet started
        # Priority queue if policy=PRIORITY, else FIFO
        self.waiting = create_request_queue(self.policy)

        # RUNNING queue: Currently generating requests
        # List because we iterate in order
        self.running: list[Request] = []

        # Finished request IDs (cleared each step)
        self.finished_req_ids: set[str] = set()

        # ================================================================
        # KV CACHE MANAGEMENT
        # ================================================================
        # WHAT: Manages physical memory blocks for PagedAttention
        # HOW: Allocates/frees blocks as requests are scheduled/finished
        # WHY: KV cache is the primary memory bottleneck

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size

        # Create the KV cache manager
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=bool(self.cache_config.enable_prefix_caching),
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
        )

        # ================================================================
        # ENCODER CACHE (for multimodal models)
        # ================================================================
        # WHAT: Separate cache for vision encoder outputs
        # WHY: Vision tokens are much larger than text KV cache

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

    def schedule(self) -> SchedulerOutput:
        """
        ====================================================================
        CORE SCHEDULING ALGORITHM: Continuous Batching
        ====================================================================

        WHAT: Decide which requests to process in this iteration
        HOW: Two-phase algorithm:
             1. Schedule RUNNING requests (already started)
             2. Add WAITING requests if resources available
        WHY: Maximize batch size while respecting resource limits

        RETURNS: SchedulerOutput containing:
        - scheduled_new_reqs: New requests starting prefill
        - scheduled_resumed_reqs: Preempted requests resuming
        - scheduled_running_reqs: Continuing requests
        - preempted_reqs: Requests that were preempted this iteration
        - req_to_new_blocks: KV cache blocks allocated
        - num_scheduled_tokens: Tokens scheduled per request

        ====================================================================
        SCHEDULING PHILOSOPHY: Generalized Token Assignment
        ====================================================================

        Traditional view: "Prefill phase" vs "Decode phase"
        vLLM view: Every request has:
        - num_computed_tokens: Tokens already processed
        - num_tokens_with_spec: Total tokens needed (prompt + output + spec)

        Each iteration: Try to close the gap between these numbers

        This generalizes to:
        - Chunked prefills (process prompt in chunks)
        - Prefix caching (skip cached prefix tokens)
        - Speculative decoding (verify spec tokens)
        - Regular decode (generate next token)

        NO EXPLICIT "PREFILL" OR "DECODE" MODES!
        """

        # Track scheduled requests by category
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # Track resource allocations
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}

        # Available resources
        token_budget = self.max_num_scheduled_tokens
        encoder_compute_budget = self.max_num_encoder_input_tokens

        # For metrics
        scheduled_timestamp = time.monotonic()

        # Spec decode tracking
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # ================================================================
        # PHASE 1: SCHEDULE RUNNING REQUESTS
        # ================================================================
        # WHAT: Try to schedule all currently running requests
        # WHY: Running requests have highest priority (they've already started)
        # HOW: Iterate through running list, allocate resources for each

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # ============================================================
            # STEP 1.1: Calculate tokens to schedule for this request
            # ============================================================
            # WHAT: How many tokens does this request need to compute?
            # HOW: num_tokens_with_spec - num_computed_tokens
            # WHY: May be more than 1 (chunked prefill, spec decode)

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            # Apply chunking if configured (long_prefill_token_threshold)
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

            # Respect token budget
            num_new_tokens = min(num_new_tokens, token_budget)

            # Don't exceed max_model_len or request's max_tokens
            max_total_tokens = min(
                request.num_prompt_tokens + request.max_tokens, self.max_model_len
            )
            num_new_tokens = min(
                num_new_tokens, max_total_tokens - 1 - request.num_computed_tokens
            )

            # ============================================================
            # STEP 1.2: Handle encoder inputs (multimodal models)
            # ============================================================
            # WHAT: Vision/audio encoder outputs for multimodal models
            # WHY: Encoder compute is separate budget from text tokens

            encoder_inputs_to_schedule = None
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                )

            if num_new_tokens == 0:
                # Cannot schedule this request (no tokens or encoder budget)
                # SKIP to next request (allows lower-priority reqs to run)
                req_index += 1
                continue

            # ============================================================
            # STEP 1.3: ALLOCATE KV CACHE BLOCKS
            # ============================================================
            # WHAT: Reserve memory blocks for this request's KV cache
            # HOW: kv_cache_manager.allocate_slots()
            # WHY: PagedAttention requires physical blocks allocated

            # CRITICAL LOOP: Try to allocate, preempt if OOM
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # Allocation successful!
                        break

                    # ================================================
                    # OUT OF MEMORY: PREEMPTION REQUIRED
                    # ================================================
                    # WHAT: No free KV cache blocks available
                    # HOW: Evict lowest-priority running request
                    # WHY: Must free memory to continue

                    if self.policy == SchedulingPolicy.PRIORITY:
                        # Find lowest-priority request (highest priority value)
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)

                        # If we already scheduled this request, undo it
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[preempted_req.request_id]
                            req_to_new_blocks.pop(preempted_req.request_id)
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(preempted_req.request_id, None)
                            # Restore encoder budget if applicable
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                num_tokens_to_restore = sum(
                                    preempted_req.get_num_encoder_tokens(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_tokens_to_restore
                            req_index -= 1
                    else:
                        # FCFS policy: Preempt last (most recent) request
                        preempted_req = self.running.pop()

                    # ================================================
                    # FREE RESOURCES FROM PREEMPTED REQUEST
                    # ================================================
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    preempted_req.num_preemptions += 1
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp
                        )

                    # Add back to waiting queue (front of line)
                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)

                    if preempted_req == request:
                        # We preempted the request we're trying to schedule!
                        # Cannot schedule any more requests
                        break

            if new_blocks is None:
                # Failed to allocate even after preemption
                break

            # ============================================================
            # STEP 1.4: SUCCESSFULLY SCHEDULED THIS REQUEST
            # ============================================================
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Handle speculative decoding tokens
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens
                )
                if num_scheduled_spec_tokens > 0:
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                request.spec_token_ids = []

            # Handle encoder inputs
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget

        # ================================================================
        # PHASE 2: SCHEDULE WAITING REQUESTS
        # ================================================================
        # WHAT: Add new requests if resources available
        # WHY: Fill remaining token budget to maximize GPU utilization
        # HOW: Pop from waiting queue, check constraints, allocate

        # Track LoRA adapters in use
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Temporary queue for requests that can't be scheduled yet
        skipped_waiting_requests = create_request_queue(self.policy)

        # Only schedule waiting requests if we didn't preempt anyone
        # (preemption means we're at capacity)
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                # Check if we hit max concurrent requests
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # ========================================================
                # CHECK 1: Waiting for remote KV cache (P/D disaggregation)
                # ========================================================
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        # Still waiting, skip for now
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # ========================================================
                # CHECK 2: Waiting for FSM compilation (structured output)
                # ========================================================
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # ========================================================
                # CHECK 3: LoRA adapter limit
                # ========================================================
                # WHAT: Limit number of different LoRA adapters per batch
                # WHY: Each adapter requires additional memory
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Would exceed max_loras, skip
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                # ========================================================
                # STEP 2.1: CHECK PREFIX CACHE
                # ========================================================
                # WHAT: Check if prompt prefix is already cached
                # HOW: kv_cache_manager.get_computed_blocks()
                # WHY: Can skip recomputing cached tokens (huge speedup!)

                num_external_computed_tokens = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    # First time scheduling this request
                    # Check for prefix cache hits
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Check external KV cache (for P/D disaggregation)
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )
                        if ext_tokens is None:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue
                        num_external_computed_tokens = ext_tokens

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # Resumed request (was preempted earlier)
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # ========================================================
                # STEP 2.2: CALCULATE TOKENS TO SCHEDULE
                # ========================================================
                if load_kv_async:
                    # Loading remote KV, don't schedule new work
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        # Chunk large prefills
                        num_new_tokens = threshold
                    num_new_tokens = min(num_new_tokens, token_budget)

                # ========================================================
                # STEP 2.3: ALLOCATE KV CACHE FOR NEW REQUEST
                # ========================================================
                if num_new_tokens > 0:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                    )

                    if new_blocks is None:
                        # Out of memory, cannot schedule more requests
                        break

                # ========================================================
                # STEP 2.4: SUCCESSFULLY SCHEDULED NEW REQUEST
                # ========================================================
                request = self.waiting.pop_request()
                self.running.append(request)

                # Update request state
                if num_computed_tokens > 0:
                    # Resumed or prefix cache hit
                    scheduled_resumed_reqs.append(request)
                    request.status = RequestStatus.RUNNING
                else:
                    # Brand new request
                    scheduled_new_reqs.append(request)
                    request.status = RequestStatus.RUNNING

                # Record allocations
                if num_new_tokens > 0:
                    req_to_new_blocks[request.request_id] = new_blocks
                    num_scheduled_tokens[request.request_id] = num_new_tokens
                    token_budget -= num_new_tokens
                else:
                    # Loading KV asynchronously
                    req_to_new_blocks[request.request_id] = new_blocks

                # Update LoRA tracking
                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)

        # ================================================================
        # RESTORE SKIPPED REQUESTS TO WAITING QUEUE
        # ================================================================
        # WHAT: Put back requests that couldn't be scheduled this iteration
        # WHY: Will try again next iteration
        while skipped_waiting_requests:
            request = skipped_waiting_requests.pop_request()
            self.waiting.prepend_request(request)

        # ================================================================
        # BUILD AND RETURN SCHEDULER OUTPUT
        # ================================================================
        # WHAT: Package all scheduling decisions for workers
        # HOW: SchedulerOutput contains batch composition + metadata
        # WHY: Workers need to know what to execute

        return SchedulerOutput(
            scheduled_new_reqs=scheduled_new_reqs,
            scheduled_resumed_reqs=scheduled_resumed_reqs,
            scheduled_running_reqs=scheduled_running_reqs,
            preempted_reqs=preempted_reqs,
            req_to_new_blocks=req_to_new_blocks,
            num_scheduled_tokens=num_scheduled_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_prefill_tokens=sum(
                num_scheduled_tokens.get(req.request_id, 0)
                for req in scheduled_new_reqs + scheduled_resumed_reqs
            ),
            num_decode_tokens=sum(
                num_scheduled_tokens.get(req.request_id, 0)
                for req in scheduled_running_reqs
            ),
            scheduled_timestamp=scheduled_timestamp,
        )


# ============================================================================
# KEY TAKEAWAYS: Scheduler & Continuous Batching
# ============================================================================
#
# 1. CONTINUOUS BATCHING ALGORITHM
#    - Phase 1: Schedule running requests (highest priority)
#    - Phase 2: Fill remaining capacity with waiting requests
#    - Dynamic batch composition changes every iteration
#
# 2. RESOURCE CONSTRAINTS
#    - Token budget: max_num_batched_tokens (prevents OOM)
#    - KV cache blocks: num_gpu_blocks (memory limit)
#    - Concurrent requests: max_num_seqs (scheduler capacity)
#    - LoRA adapters: max_loras (multi-tenant limit)
#
# 3. PREEMPTION MECHANISM
#    - When out of KV cache: Evict lowest-priority request
#    - Preempted request returns to front of waiting queue
#    - Loses all computed KV cache (must restart from beginning)
#    - Tracked via num_preemptions metric
#
# 4. PREFIX CACHING
#    - Check computed_blocks before scheduling new request
#    - Can skip recomputing cached prefix tokens
#    - Massive speedup for requests with shared prompts
#    - Implemented via kv_cache_manager.get_computed_blocks()
#
# 5. NO EXPLICIT PREFILL/DECODE PHASES
#    - Every request just has: num_computed_tokens vs num_tokens_with_spec
#    - Scheduler tries to close this gap each iteration
#    - Naturally handles: chunked prefill, prefix caching, spec decode
#
# 6. FAIRNESS
#    - RUNNING requests always prioritized over WAITING
#    - Within same priority: FCFS (first-come-first-served)
#    - Prevents starvation of long-running requests
#
# 7. PERFORMANCE OPTIMIZATION
#    - Skip requests that can't be scheduled (allows lower-priority to run)
#    - Batch prefill and decode together (mixed batching)
#    - Apply chunking to large prefills (long_prefill_token_threshold)
#    - Track LoRA adapters to respect max_loras constraint
#
# ============================================================================
