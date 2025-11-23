# ANNOTATED: vLLM LLMEngine - Core Request Orchestration Engine
# Source: vllm/v1/engine/llm_engine.py
#
# ============================================================================
# WHAT: The LLMEngine is the primary user-facing orchestrator for inference
# HOW: Coordinates between input processing, scheduling, execution, and output
# WHY: Provides a clean abstraction layer separating user requests from low-level
#      execution details (batching, memory management, distributed compute)
# ============================================================================

"""
ARCHITECTURAL OVERVIEW:

The LLMEngine follows a pipeline architecture:

1. REQUEST INTAKE (add_request)
   User Request → Processor → EngineCoreRequest

2. EXECUTION LOOP (step)
   EngineCore.get_output() → OutputProcessor → RequestOutput

3. MEMORY MANAGEMENT
   Handled by EngineCore (Scheduler + KVCacheManager)

Key Design Principles:
- Separation of concerns: Engine handles orchestration, not execution
- Async-ready: All operations designed to support async/await patterns
- Multi-process capable: Can run in single-process or multi-process mode
- Data parallelism support: Coordinates across multiple GPUs/TP ranks
"""

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any, cast

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer import AnyTokenizer, init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.engine.processor import Processor
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """
    ========================================================================
    CORE ORCHESTRATOR: LLMEngine
    ========================================================================

    WHAT: The main serving engine that processes user requests end-to-end

    HOW:
    - Tokenizes input text via Processor
    - Delegates scheduling/execution to EngineCore
    - Detokenizes output via OutputProcessor
    - Manages lifecycle of requests (add, step, abort)

    WHY: Provides a simple, stateful API for serving that hides complexity:
    - Users call add_request() and step()
    - Engine handles batching, memory, distributed execution internally

    CRITICAL COMPONENTS:
    1. Processor: Tokenizes inputs, handles multimodal data
    2. EngineCore: Runs scheduling + model execution loop
    3. OutputProcessor: Detokenizes and formats outputs
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None:
        """
        INITIALIZATION FLOW:

        STEP 1: Store configurations
        - vllm_config contains ALL configuration (model, cache, scheduler, etc.)
        - These configs are used throughout the engine lifecycle

        STEP 2: Initialize Data Parallelism (if needed)
        - Data parallel allows serving same model across multiple GPU ranks
        - Each rank processes different requests independently
        - Important: Must init DP group BEFORE EngineCore

        STEP 3: Create the processing pipeline
        - Processor: Handles tokenization and multimodal inputs
        - OutputProcessor: Handles detokenization and streaming
        - EngineCore: The actual inference execution engine

        WHY THIS ORDER?
        - DP group must exist before workers initialize
        - Processor needs tokenizer before requests arrive
        - EngineCore depends on all configs being set
        """
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.log_stats = log_stats

        # ====================================================================
        # DATA PARALLELISM SETUP
        # ====================================================================
        # WHAT: Data parallelism (DP) replicates the model across GPUs
        # HOW: Each DP rank is an independent serving instance
        # WHY: Increase throughput by processing multiple requests in parallel
        #      Different from tensor parallelism (TP) which splits model weights

        executor_backend = self.vllm_config.parallel_config.distributed_executor_backend
        parallel_config = vllm_config.parallel_config
        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )

        # Initialize DP process group if needed
        # NOTE: This must happen BEFORE EngineCoreClient.make_client()
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        # ====================================================================
        # TOKENIZER INITIALIZATION
        # ====================================================================
        # WHAT: Tokenizer converts text → token IDs and vice versa
        # HOW: Loaded from HuggingFace tokenizer configs
        # WHY: Required for all text generation tasks
        #      Can be skipped for embedding models or when using external tokenization

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = init_tokenizer_from_configs(self.model_config)

        # ====================================================================
        # INPUT/OUTPUT PROCESSORS
        # ====================================================================
        # WHAT: Processor handles request preprocessing (tokenization, etc.)
        #       OutputProcessor handles postprocessing (detokenization, streaming)
        # HOW: Processor → tokenizes, validates, prepares EngineCoreRequest
        #      OutputProcessor → detokenizes, formats RequestOutput
        # WHY: Separates text processing from core inference logic

        self.processor = Processor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor: Converts engine outputs → user-facing RequestOutput
        # stream_interval controls how often we send intermediate tokens in streaming mode
        stream_interval = self.vllm_config.scheduler_config.stream_interval
        self.output_processor = OutputProcessor(
            self.tokenizer, log_stats=self.log_stats, stream_interval=stream_interval
        )

        # Optional: OpenTelemetry tracing for observability
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # ====================================================================
        # ENGINE CORE: The Heart of Inference
        # ====================================================================
        # WHAT: EngineCore runs the core scheduling + execution loop
        # HOW: Continuously:
        #      1. Scheduler decides which requests to process
        #      2. Workers execute model forward pass
        #      3. Returns outputs to LLMEngine
        # WHY: Separates high-level orchestration (LLMEngine) from
        #      low-level execution (EngineCore)
        #
        # MULTIPROCESS MODE:
        # - False: EngineCore runs in same process (lower latency, simpler)
        # - True: EngineCore runs in separate process (better isolation)

        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,  # Synchronous mode (for async, use AsyncLLM)
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # ====================================================================
        # METRICS AND LOGGING
        # ====================================================================
        # WHAT: StatLoggerManager collects and logs performance metrics
        # HOW: Records throughput, latency, cache hit rates, etc.
        # WHY: Critical for monitoring production deployments

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # For v0 compatibility: expose model_executor directly
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # Reuse existing DP group for external launcher mode
            self.dp_group = get_dp_group().cpu_group

        # Clear any dummy multimodal cache data from initialization
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        """
        FACTORY METHOD: Create LLMEngine from VllmConfig

        WHAT: Alternative constructor using VllmConfig directly
        HOW: Determines executor class from config, then calls __init__
        WHY: Cleaner API when you already have a VllmConfig object
        """
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """
        FACTORY METHOD: Create LLMEngine from CLI/API arguments

        WHAT: Most common way to create an LLMEngine
        HOW:
        1. EngineArgs → VllmConfig (validates and normalizes config)
        2. VllmConfig → Executor class (based on hardware/parallelism)
        3. Create LLMEngine with validated config

        WHY: This is the entry point used by:
        - vllm serve CLI command
        - OpenAI API server
        - LLM() high-level API

        EXAMPLE:
        ```python
        engine_args = EngineArgs(
            model="meta-llama/Llama-2-7b-hf",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9
        )
        engine = LLMEngine.from_engine_args(engine_args)
        ```
        """
        # Create the engine configs from arguments
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def get_num_unfinished_requests(self) -> int:
        """
        WHAT: Returns count of requests still being processed
        HOW: Queries OutputProcessor's internal request tracking
        WHY: Used to check if engine is idle (important for shutdown)
        """
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """
        WHAT: Checks if any requests are still in-flight
        HOW: Checks both OutputProcessor and EngineCore
        WHY:
        - OutputProcessor: Tracks requests this rank knows about
        - EngineCore: May have requests from other DP ranks
        - Both must be empty before safe shutdown

        DATA PARALLELISM NOTE:
        In DP mode, must synchronize across all ranks to determine
        if the entire system has unfinished requests
        """
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            # No data parallelism: just check local state
            return has_unfinished or self.engine_core.dp_engines_running()
        # With DP: must aggregate across all ranks
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        """
        DATA PARALLELISM COORDINATION

        WHAT: Synchronizes "has_unfinished" state across DP ranks
        HOW: Uses AllReduce to check if ANY rank has unfinished requests
        WHY: In DP mode, one rank may finish while others are still working

        EDGE CASE: Dummy batches
        If other ranks have work but this rank doesn't, this rank must
        execute "dummy batches" to stay synchronized (required for NCCL)
        """
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            # This rank is idle but others aren't → execute dummy batch
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def abort_request(self, request_ids: list[str]) -> None:
        """
        REQUEST CANCELLATION

        WHAT: Immediately stops processing specified requests
        HOW:
        1. Remove from OutputProcessor (stops tracking outputs)
        2. Remove from EngineCore (frees KV cache, removes from scheduler)

        WHY: Users may cancel requests (e.g., HTTP client disconnects)

        CRITICAL: Must abort in BOTH places to fully free resources
        """
        # OutputProcessor may filter the list (e.g., if already finished)
        request_ids = self.output_processor.abort_requests(request_ids)
        # Tell EngineCore to free KV cache and remove from scheduler
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> None:
        """
        ====================================================================
        REQUEST INTAKE: Adding a New Generation Request
        ====================================================================

        WHAT: Registers a new request for text generation or pooling
        HOW:
        1. Tokenize the prompt (if raw text/dict provided)
        2. Create internal RequestState in OutputProcessor
        3. Queue request to EngineCore for scheduling

        WHY: This is the main entry point for all inference requests

        FLOW:
        add_request() → Processor.process_inputs() → EngineCoreRequest
                     → OutputProcessor.add_request() (track state)
                     → EngineCore.add_request() (schedule execution)

        PARAMETERS:
        - request_id: Unique identifier (often UUID)
        - prompt: Either pre-tokenized EngineCoreRequest OR raw text/dict
        - params: SamplingParams (text gen) or PoolingParams (embeddings)
        - priority: Higher values = higher scheduling priority
        - lora_request: Optional LoRA adapter to apply

        PARALLEL SAMPLING (params.n > 1):
        When user requests multiple completions for same prompt:
        - Create N child requests (one per completion)
        - Each child gets unique request_id
        - ParentRequest aggregates child outputs
        """

        # Validate request_id type
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # ================================================================
        # STEP 1: TOKENIZATION
        # ================================================================
        # If prompt is already an EngineCoreRequest, skip tokenization
        # Otherwise, use Processor to convert text → tokens

        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            assert prompt_text is None
            logger.warning_once(
                "Processor has been moved under LLM and will "
                "be removed from LLMEngine in v0.13."
            )
            # Process inputs: tokenize, validate, create EngineCoreRequest
            request = self.processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
            )
            # Extract prompt text for logging/debugging
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        # ================================================================
        # STEP 2: HANDLE PARALLEL SAMPLING (n > 1)
        # ================================================================
        # WHAT: Generate multiple completions for the same prompt
        # HOW: Create N child requests, each with unique ID
        # WHY: Common for chat applications (generate multiple responses)

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Simple case: Single completion
            # Add to OutputProcessor for tracking
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Queue to EngineCore for execution
            self.engine_core.add_request(request)
            return

        # ================================================================
        # PARALLEL SAMPLING: Fan out to N child requests
        # ================================================================
        # WHAT: Create N independent requests, one per completion
        # HOW: ParentRequest manages child IDs and aggregates outputs
        # WHY: Scheduler treats each child as independent request

        parent_req = ParentRequest(request_id, params)
        for idx in range(n):
            # Get child request ID and modified params
            request_id, params = parent_req.get_child_info(idx)

            # Reuse same request object for all but last child (optimization)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = params

            # Register child with OutputProcessor
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Queue child to EngineCore
            self.engine_core.add_request(child_request)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """
        ====================================================================
        EXECUTION STEP: Process One Iteration
        ====================================================================

        WHAT: Executes one forward pass for all scheduled requests
        HOW:
        1. Get outputs from EngineCore (one generation step)
        2. Process outputs (detokenize, check stop conditions)
        3. Abort any requests that finished
        4. Log metrics

        WHY: This is the main serving loop. Called repeatedly to:
        - Generate next tokens for all active requests
        - Return completed/partial outputs to users

        CRITICAL FLOW:
        ```
        step() → EngineCore.get_output() → Scheduler.schedule()
                                          → Workers.execute_model()
                                          → Sampler.sample()
               → OutputProcessor.process_outputs()
               → Return RequestOutput[]
        ```

        RETURNS:
        List of RequestOutput objects, one per request that has new output

        DATA PARALLELISM NOTE:
        In DP mode, may need to execute dummy batch to stay synchronized
        with other DP ranks
        """

        # ================================================================
        # DATA PARALLELISM: Handle dummy batches
        # ================================================================
        # WHAT: Execute empty forward pass to synchronize with other ranks
        # WHY: NCCL collectives require all ranks to participate
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []  # No actual outputs

        # ================================================================
        # STEP 1: GET ENGINE OUTPUTS
        # ================================================================
        # WHAT: Get results from one scheduling + execution iteration
        # HOW: EngineCore internally:
        #      1. Scheduler decides which requests to process
        #      2. Workers run model forward pass
        #      3. Sampler generates next tokens
        # WHY: This is where the actual inference happens

        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # ================================================================
        # STEP 2: PROCESS OUTPUTS
        # ================================================================
        # WHAT: Convert raw outputs → user-facing RequestOutput objects
        # HOW: OutputProcessor:
        #      - Detokenizes token IDs → text
        #      - Checks stop conditions (stop strings, max_tokens, etc.)
        #      - Formats streaming outputs
        # WHY: EngineCore returns raw tokens, users need text + metadata

        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            # Update scheduler statistics for monitoring
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # ================================================================
        # STEP 3: ABORT FINISHED REQUESTS
        # ================================================================
        # WHAT: Clean up requests that hit stop conditions
        # HOW: OutputProcessor identified finished requests during processing
        # WHY: Must free KV cache and remove from scheduler queue

        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # ================================================================
        # STEP 4: RECORD METRICS
        # ================================================================
        # WHAT: Log throughput, latency, cache stats, etc.
        # WHY: Critical for monitoring production deployments

        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if self.logger_manager is not None and outputs.scheduler_stats is not None:
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.processor.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self):
        """Start PyTorch profiler for performance analysis"""
        self.engine_core.profile(True)

    def stop_profile(self):
        """Stop PyTorch profiler"""
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        """Clear multimodal (vision/audio) preprocessing cache"""
        self.processor.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self):
        """
        WHAT: Invalidate all prefix cache entries
        WHY: Required after model weight updates (e.g., during RLHF training)
        """
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1):
        """
        POWER MANAGEMENT: Put engine to sleep

        WHAT: Reduce power consumption during idle periods
        HOW: Releases GPU memory, pauses workers
        WHY: Cost optimization for serverless deployments
        """
        self.engine_core.sleep(level)
        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        """Wake engine from sleep state"""
        self.engine_core.wake_up(tags)
        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        """Check if engine is in sleep state"""
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        """Get current performance metrics snapshot"""
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> AnyTokenizer | None:
        return self.processor.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: AnyTokenizer | None) -> None:
        self.processor.tokenizer = tokenizer

    def get_tokenizer(self) -> AnyTokenizer:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because skip_tokenizer_init is True"
            )
        return self.tokenizer

    def do_log_stats(self) -> None:
        """Immediately log all accumulated stats"""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """
        PERIODIC LOGGING

        WHAT: Log stats only if enough time has passed since last log
        WHY: Avoid spamming logs, but ensure regular updates
        """
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    # ====================================================================
    # LORA MANAGEMENT
    # ====================================================================

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """
        WHAT: Load a LoRA adapter for future requests
        HOW: Downloads weights, registers with model executor
        WHY: Support multi-tenant serving with different fine-tuned adapters
        """
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Unload a LoRA adapter to free memory"""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """Get all currently loaded LoRA adapter IDs"""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """
        WHAT: Prevent a LoRA adapter from being evicted
        WHY: Frequently-used adapters should stay in memory
        """
        return self.engine_core.pin_lora(lora_id)

    # ====================================================================
    # ADVANCED: Direct Worker Access
    # ====================================================================

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        """
        ADVANCED: Execute arbitrary function on all workers

        WHAT: Call a method on all workers and collect results
        WHY: For advanced use cases (custom ops, debugging, etc.)
        """
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        """
        ADVANCED: Apply function to model on all workers

        EXAMPLE:
        ```python
        def get_layer_count(model):
            return len(model.layers)

        counts = engine.apply_model(get_layer_count)
        ```
        """
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        """Cleanup: Destroy data parallel process group"""
        if (
            dp_group := getattr(self, "dp_group", None)
            and not self.external_launcher_dp
        ):
            stateless_destroy_torch_distributed_process_group(dp_group)


# ============================================================================
# KEY TAKEAWAYS: LLMEngine
# ============================================================================
#
# 1. ORCHESTRATION, NOT EXECUTION
#    LLMEngine coordinates components but doesn't do heavy lifting itself
#    Actual inference happens in EngineCore → Scheduler → Workers
#
# 2. CLEAN ABSTRACTION
#    Users interact with simple API: add_request(), step(), abort_request()
#    Engine hides complexity of batching, memory, distributed execution
#
# 3. PIPELINE ARCHITECTURE
#    Request flow: add_request() → Processor → EngineCore → OutputProcessor
#    Each component has clear responsibility
#
# 4. DATA PARALLELISM SUPPORT
#    Can coordinate multiple DP ranks for higher throughput
#    Handles synchronization and dummy batches
#
# 5. PRODUCTION READY
#    - Comprehensive metrics and logging
#    - Request cancellation
#    - Power management (sleep/wake)
#    - LoRA multi-tenancy
#
# ============================================================================
