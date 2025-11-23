# Inference Request Lifecycle in vLLM: Complete Data Flow Guide

## Table of Contents
1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Request Lifecycle Phases](#request-lifecycle-phases)
4. [Detailed Phase Breakdown](#detailed-phase-breakdown)
5. [Memory Flow: KV Cache Lifecycle](#memory-flow-kv-cache-lifecycle)
6. [Streaming Response Flow](#streaming-response-flow)
7. [Error Handling and Edge Cases](#error-handling-and-edge-cases)
8. [Performance Metrics](#performance-metrics)

---

## Overview

This guide traces a **single streaming inference request** from HTTP POST to streamed tokens, explaining:
- How the request flows through vLLM's components
- When and how KV cache is allocated/deallocated
- How continuous batching integrates new requests
- How responses are streamed back to the user

**Example Request:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": true
  }'
```

**Request ID**: `req-123abc` (auto-generated UUID)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER (HTTP CLIENT)                          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ POST /v1/completions
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   FASTAPI SERVER (api_server.py)                    │
│  - Parse HTTP request                                               │
│  - Validate JSON schema                                             │
│  - Create OpenAI CompletionRequest object                           │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ CompletionRequest
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│             SERVING ENGINE (serving_engine.py)                      │
│  - Convert CompletionRequest → EngineCoreRequest                    │
│  - Call LLMEngine.add_request()                                     │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ EngineCoreRequest
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   PROCESSOR (processor.py)                          │
│  - Tokenize prompt text → token IDs                                │
│  - Create Request object with metadata                              │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Request
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 OUTPUT PROCESSOR (output_processor.py)              │
│  - Create RequestState for tracking                                │
│  - Initialize output buffers                                        │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ RequestState created
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   ENGINE CORE (core.py)                             │
│  - Add request to Scheduler's WAITING queue                         │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
                  ↓
        ╔═════════════════════════════════╗
        ║   CONTINUOUS BATCHING LOOP      ║
        ║   (Runs every iteration)        ║
        ╚═════════════════════════════════╝
                  │
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   SCHEDULER (scheduler.py)                          │
│  - Decide which requests to process this iteration                 │
│  - Allocate KV cache blocks via KVCacheManager                     │
│  - Create SchedulerOutput (batch composition)                      │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ SchedulerOutput
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│               EXECUTOR (executor/gpu_executor.py)                   │
│  - Distribute batch to GPU workers                                 │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Batch metadata
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              GPU WORKER (worker/gpu_worker.py)                      │
│  - Build InputBatch for model                                      │
│  - Prepare BlockTable (logical → physical mapping)                 │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ InputBatch
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│           GPU MODEL RUNNER (worker/gpu_model_runner.py)             │
│  - Run model.forward(input_ids, positions, kv_caches)              │
│  - PagedAttention kernel reads/writes KV cache                     │
│  - Get logits for next token                                       │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Logits (shape: [num_tokens, vocab_size])
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   SAMPLER (sample/sampler.py)                       │
│  - Apply temperature, top-p, top-k                                 │
│  - Sample next token ID                                            │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Token IDs
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              OUTPUT PROCESSOR (output_processor.py)                 │
│  - Detokenize token IDs → text                                     │
│  - Check stop conditions (stop strings, max_tokens)                │
│  - Format RequestOutput                                            │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ RequestOutput
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│              SERVING ENGINE (serving_engine.py)                     │
│  - Convert RequestOutput → CompletionResponse                      │
│  - Stream chunk to client                                          │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ Server-Sent Event (SSE)
                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         USER (HTTP CLIENT)                          │
│  - Receive: data: {"choices": [{"text": "Quantum"}]}               │
└─────────────────────────────────────────────────────────────────────┘
```

**Loop continues** until request finishes (stop string, max_tokens, EOS)

---

## Request Lifecycle Phases

### Phase 1: Request Intake (HTTP → Internal Request)
**Duration**: ~1-5 ms
**Components**: FastAPI → ServingEngine → Processor

### Phase 2: Waiting (Queue)
**Duration**: Variable (0 ms if resources available, or until scheduled)
**Components**: Scheduler WAITING queue

### Phase 3: Prefill (First Token)
**Duration**: ~50-500 ms (depends on prompt length)
**Components**: Scheduler → Worker → Model → Sampler
**What Happens**: Entire prompt processed in parallel, KV cache populated

### Phase 4: Decode (Subsequent Tokens)
**Duration**: ~10-30 ms per token
**Components**: Same as prefill, but processes 1 token at a time
**Iterations**: Repeats until completion (e.g., 100 iterations for 100 tokens)

### Phase 5: Completion & Cleanup
**Duration**: ~1-2 ms
**Components**: OutputProcessor → KVCacheManager (free blocks)

---

## Detailed Phase Breakdown

### PHASE 1: REQUEST INTAKE

#### Step 1.1: HTTP Request Arrives
**File**: `vllm/entrypoints/openai/api_server.py:create_completion()`

```
User sends:
POST /v1/completions
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "prompt": "Explain quantum computing in simple terms:",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

**What Happens**:
1. FastAPI validates JSON schema
2. Creates `CompletionRequest` object
3. Calls `serving_engine.create_completion()`

#### Step 1.2: Request Conversion
**File**: `vllm/entrypoints/openai/serving_engine.py:create_completion()`

```python
# Convert HTTP request → SamplingParams
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=100,
    # ... other params
)

# Generate request ID
request_id = f"req-{uuid.uuid4()}"  # "req-123abc"

# Add to engine
await engine.add_request(
    request_id=request_id,
    prompt=prompt_text,
    params=sampling_params,
)
```

**Output**: `request_id = "req-123abc"`

#### Step 1.3: Tokenization
**File**: `vllm/v1/engine/processor.py:process_inputs()`

```python
# Tokenize prompt
token_ids = tokenizer.encode("Explain quantum computing in simple terms:")
# Result: [1, 12027, 15150, 20795, 297, 2560, 4958, 29901]  # 8 tokens

# Create EngineCoreRequest
request = EngineCoreRequest(
    request_id="req-123abc",
    prompt_token_ids=token_ids,
    num_prompt_tokens=8,
    max_tokens=100,
    sampling_params=sampling_params,
    arrival_time=time.monotonic(),
)
```

**Key Metrics**:
- Prompt length: 8 tokens
- Max total length: 8 + 100 = 108 tokens
- VRAM needed: ~108 × 2 × 32 layers × 4096 hidden_dim × 2 bytes ≈ 55 MB per request

#### Step 1.4: Create Request State
**File**: `vllm/v1/engine/output_processor.py:add_request()`

```python
# Create state tracker
request_state = RequestState(
    request_id="req-123abc",
    prompt_text="Explain quantum computing in simple terms:",
    output_text="",  # Empty, will accumulate tokens
    num_computed_tokens=0,  # No tokens processed yet
    num_output_tokens=0,
)

self.request_states["req-123abc"] = request_state
```

#### Step 1.5: Queue to Scheduler
**File**: `vllm/v1/engine/core.py:add_request()`

```python
# Create Request object
request = Request(
    request_id="req-123abc",
    prompt_token_ids=[1, 12027, 15150, 20795, 297, 2560, 4958, 29901],
    num_prompt_tokens=8,
    max_tokens=100,
    sampling_params=sampling_params,
    arrival_time=time.monotonic(),
    status=RequestStatus.WAITING,
)

# Add to scheduler's WAITING queue
scheduler.waiting.push(request)
```

**Status**: Request is now **WAITING** for scheduling

---

### PHASE 2: WAITING IN QUEUE

**File**: `vllm/v1/core/sched/scheduler.py:schedule()`

```python
# Scheduler loop (runs every iteration)
while True:
    # Check if we can schedule this request
    if scheduler.can_schedule(request):
        # Allocate KV cache blocks
        blocks = kv_cache_manager.allocate_slots(request, num_tokens=8)
        if blocks is not None:
            # Success! Move to RUNNING
            request.status = RequestStatus.RUNNING
            scheduler.running.append(request)
            scheduler.waiting.remove(request)
            break
    # Else: Wait for next iteration
    await asyncio.sleep(0.001)  # 1ms polling
```

**Typical Wait Time**:
- If server is idle: **0 ms** (scheduled immediately)
- If server is busy: **10-1000 ms** (depends on batch size and load)

---

### PHASE 3: PREFILL (First Forward Pass)

#### Step 3.1: Scheduler Decides to Process Request
**File**: `vllm/v1/core/sched/scheduler.py:schedule()`

```python
# Iteration 1: Prefill
scheduler_output = scheduler.schedule()
# Returns:
# SchedulerOutput(
#     scheduled_new_reqs=[Request("req-123abc")],  # New request!
#     num_scheduled_tokens={"req-123abc": 8},  # All 8 prompt tokens
#     req_to_new_blocks={"req-123abc": KVCacheBlocks([block_42])},
# )
```

**What Happened**:
- Request selected from WAITING queue
- KV cache block allocated: Physical block #42
- All 8 prompt tokens scheduled for processing

#### Step 3.2: Allocate KV Cache Blocks
**File**: `vllm/v1/core/kv_cache_manager.py:allocate_slots()`

```python
# Calculate blocks needed
num_blocks_needed = ceil(8 tokens / 16 block_size) = 1 block

# Allocate from block pool
block = block_pool.allocate()  # Returns block #42
request.kv_blocks = [block_42]

# Return allocation
return KVCacheBlocks([block_42])
```

**Memory State**:
```
KV Cache (GPU VRAM):
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │  ...    │ Block 42│ ← req-123abc
│ (Free)  │ (Free)  │         │ (Alloc) │
└─────────┴─────────┴─────────┴─────────┘
```

#### Step 3.3: Build Input Batch
**File**: `vllm/v1/worker/gpu_worker.py:prepare_input()`

```python
# Prepare inputs for model
input_batch = InputBatch(
    req_ids=["req-123abc"],
    prompt_token_ids=[1, 12027, 15150, 20795, 297, 2560, 4958, 29901],
    positions=[0, 1, 2, 3, 4, 5, 6, 7],  # Token positions
    num_prefill_tokens=8,
    num_decode_tokens=0,  # No decode yet
)
```

#### Step 3.4: Compute Slot Mapping
**File**: `vllm/v1/worker/block_table.py:compute_slot_mapping()`

```python
# Map token positions → physical memory slots
req_indices = [0, 0, 0, 0, 0, 0, 0, 0]  # All from request 0
positions = [0, 1, 2, 3, 4, 5, 6, 7]

# block_table[0] = [42]  (request 0 uses block 42)
# For each token:
#   block_idx = position // block_size = [0, 0, 0, ..., 0]
#   physical_block = block_table[req_idx][block_idx] = [42, 42, ..., 42]
#   offset = position % block_size = [0, 1, 2, 3, 4, 5, 6, 7]
#   slot = physical_block × block_size + offset

slot_mapping = [
    42 × 16 + 0 = 672,
    42 × 16 + 1 = 673,
    42 × 16 + 2 = 674,
    42 × 16 + 3 = 675,
    42 × 16 + 4 = 676,
    42 × 16 + 5 = 677,
    42 × 16 + 6 = 678,
    42 × 16 + 7 = 679,
]
```

**Result**: Tokens will write KV cache to slots 672-679

#### Step 3.5: Model Forward Pass (Prefill)
**File**: `vllm/v1/worker/gpu_model_runner.py:execute_model()`

```python
# Run transformer model
with torch.no_grad():
    logits = model.forward(
        input_ids=torch.tensor([1, 12027, 15150, 20795, 297, 2560, 4958, 29901]),
        positions=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        kv_caches=kv_cache_tensors,  # GPU memory for KV cache
        slot_mapping=torch.tensor([672, 673, 674, 675, 676, 677, 678, 679]),
    )
# Output: logits.shape = [8, 32000]  # 8 tokens, 32K vocab
```

**What Happens Inside**:
1. Embedding layer: token_ids → embeddings
2. For each transformer layer (32 layers):
   - **Attention**: Compute Q, K, V
   - **PagedAttention kernel**:
     - **Write** K, V to slots 672-679 (physical block 42)
     - **Read** K, V from slots 672-679
     - Compute attention scores
   - **MLP**: Feed-forward network
3. Final layer: hidden states → logits

**KV Cache State After Prefill**:
```
Block 42 (slots 672-679):
┌────┬────┬────┬────┬────┬────┬────┬────┬─────┬─────┐
│ K0 │ K1 │ K2 │ K3 │ K4 │ K5 │ K6 │ K7 │ ... │ ... │
│ V0 │ V1 │ V2 │ V3 │ V4 │ V5 │ V6 │ V7 │ ... │ ... │
└────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┘
  ↑                                          ↑
  Slot 672                                   Slot 679
```

#### Step 3.6: Sample First Token
**File**: `vllm/v1/sample/sampler.py:sample()`

```python
# Get logits for last token (position 7)
last_token_logits = logits[7, :]  # Shape: [32000]

# Apply temperature
logits = last_token_logits / temperature  # 0.7

# Apply top-p (nucleus sampling)
sorted_logits, sorted_indices = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)
top_p_mask = cumulative_probs <= 0.9  # top_p=0.9
filtered_logits = logits[top_p_mask]

# Sample
probs = softmax(filtered_logits)
next_token_id = torch.multinomial(probs, num_samples=1)
# Result: next_token_id = 12278  # "Quantum"
```

#### Step 3.7: Detokenize and Stream
**File**: `vllm/v1/engine/output_processor.py:process_outputs()`

```python
# Detokenize token ID → text
next_token_text = tokenizer.decode([12278])  # "Quantum"

# Update request state
request_state.output_text += next_token_text  # "" + "Quantum" = "Quantum"
request_state.num_output_tokens += 1  # 0 + 1 = 1
request.num_computed_tokens = 8  # Prompt fully processed

# Create output
output = RequestOutput(
    request_id="req-123abc",
    prompt="Explain quantum computing in simple terms:",
    outputs=[CompletionOutput(text="Quantum")],
    finished=False,
)

return output
```

#### Step 3.8: Stream to Client
**File**: `vllm/entrypoints/openai/serving_engine.py`

```python
# Format as Server-Sent Event
response_chunk = {
    "id": "req-123abc",
    "object": "text_completion",
    "created": 1234567890,
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "choices": [{
        "text": "Quantum",
        "index": 0,
        "finish_reason": None,
    }],
}

# Stream to client
yield f"data: {json.dumps(response_chunk)}\n\n"
```

**Client Receives**:
```
data: {"id":"req-123abc","object":"text_completion","choices":[{"text":"Quantum","index":0,"finish_reason":null}]}
```

---

### PHASE 4: DECODE (Subsequent Tokens)

**Iterations 2-100** (Generate remaining 99 tokens)

#### Step 4.1: Schedule Decode Token
**File**: `vllm/v1/core/sched/scheduler.py:schedule()`

```python
# Iteration 2: Decode
request.num_computed_tokens = 8  # Prefill done
request.num_tokens = 9  # Prompt (8) + output (1)

num_new_tokens = request.num_tokens - request.num_computed_tokens  # 9 - 8 = 1

# Allocate 1 more KV cache slot
# Block 42 has 16 slots, 8 used → 8 free
# No new block needed!
new_blocks = None  # Reuse existing block
```

#### Step 4.2: Compute Slot Mapping (Decode)
**File**: `vllm/v1/worker/block_table.py:compute_slot_mapping()`

```python
# For token at position 8 (first generated token):
req_indices = [0]
positions = [8]

# Calculation:
# block_idx = 8 // 16 = 0 → Still block 42
# offset = 8 % 16 = 8
# slot = 42 × 16 + 8 = 680

slot_mapping = [680]
```

#### Step 4.3: Model Forward Pass (Decode)
**File**: `vllm/v1/worker/gpu_model_runner.py:execute_model()`

```python
# Run model for SINGLE token
logits = model.forward(
    input_ids=torch.tensor([12278]),  # Last generated token: "Quantum"
    positions=torch.tensor([8]),      # Position in sequence
    kv_caches=kv_cache_tensors,
    slot_mapping=torch.tensor([680]), # Write K,V to slot 680
)
# Output: logits.shape = [1, 32000]  # 1 token, 32K vocab
```

**PagedAttention Kernel**:
- **Write** K, V for token "Quantum" to slot 680
- **Read** K, V for ALL previous tokens (slots 672-680)
- Compute attention with 9 tokens total

**KV Cache State After Iteration 2**:
```
Block 42 (slots 672-687):
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬─────┬─────┐
│ K0 │ K1 │ K2 │ K3 │ K4 │ K5 │ K6 │ K7 │ K8 │ ... │ ... │
│ V0 │ V1 │ V2 │ V3 │ V4 │ V5 │ V6 │ V7 │ V8 │ ... │ ... │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴─────┴─────┘
  ↑                                      ↑
  Slot 672                               Slot 680
```

#### Step 4.4: Sample Next Token
```python
# Sample next token
next_token_id = sample(logits[0])  # e.g., 20795 → "computing"
```

#### Step 4.5: Stream to Client
```python
# Detokenize
next_token_text = tokenizer.decode([20795])  # " computing"
request_state.output_text += next_token_text  # "Quantum computing"

# Stream
yield f'data: {{"choices":[{{"text":" computing"}}]}}\n\n'
```

**This repeats for 98 more iterations** until 100 tokens generated.

---

### PHASE 5: COMPLETION & CLEANUP

#### Step 5.1: Detect Stop Condition
**File**: `vllm/v1/engine/output_processor.py:process_outputs()`

```python
# Check if finished
if request.num_output_tokens >= request.max_tokens:  # 100 >= 100
    request.finished = True
    finish_reason = "length"

# Or if EOS token generated
if next_token_id == tokenizer.eos_token_id:
    request.finished = True
    finish_reason = "stop"

# Or if stop string found
if any(stop_str in request_state.output_text for stop_str in stop_strings):
    request.finished = True
    finish_reason = "stop"
```

#### Step 5.2: Send Final Chunk
```python
# Final streaming chunk
response_chunk = {
    "id": "req-123abc",
    "choices": [{
        "text": "",  # No new text
        "finish_reason": "length",  # Stopped due to max_tokens
    }],
}

yield f"data: {json.dumps(response_chunk)}\n\n"
yield "data: [DONE]\n\n"  # Signal stream end
```

#### Step 5.3: Free KV Cache
**File**: `vllm/v1/core/kv_cache_manager.py:free()`

```python
# Free allocated blocks
kv_cache_manager.free(request)

# Internally:
# for block in request.kv_blocks:  # [block_42]
#     block_pool.free(block_42)

# Block 42 now available for reuse
```

**Memory State After Cleanup**:
```
KV Cache (GPU VRAM):
┌─────────┬─────────┬─────────┬─────────┐
│ Block 0 │ Block 1 │  ...    │ Block 42│
│ (Free)  │ (Free)  │         │ (Free)  │ ← Available again!
└─────────┴─────────┴─────────┴─────────┘
```

#### Step 5.4: Remove from Request States
```python
# Clean up tracking
del output_processor.request_states["req-123abc"]
del scheduler.requests["req-123abc"]

# Request fully cleaned up
```

---

## Memory Flow: KV Cache Lifecycle

### Timeline of KV Cache Usage

```
Time: 0ms (Request arrives)
┌──────────────────────────┐
│ KV Cache: 0 blocks used  │
└──────────────────────────┘

Time: 10ms (Prefill starts)
┌──────────────────────────────────┐
│ Allocate: 1 block (block 42)     │
│ Usage: 8/16 slots in block 42    │
└──────────────────────────────────┘

Time: 50ms (Prefill complete)
┌──────────────────────────────────┐
│ Usage: 8/16 slots (prompt KV)    │
└──────────────────────────────────┘

Time: 60ms - 3000ms (Decode 100 tokens)
┌──────────────────────────────────────────┐
│ Iteration 1: 9/16 slots                  │
│ Iteration 2: 10/16 slots                 │
│ ...                                      │
│ Iteration 8: 16/16 slots (block full!)   │
│ Allocate: 1 more block (block 17)       │
│ Iteration 9: 1/16 slots in block 17      │
│ ...                                      │
│ Iteration 100: 8/16 slots in block 17    │
│                                          │
│ Total: 2 blocks (42, 17)                 │
│ Total slots: 108/32 (8 prompt + 100 gen)│
└──────────────────────────────────────────┘

Time: 3100ms (Request finishes)
┌──────────────────────────┐
│ Free blocks: 42, 17      │
│ KV Cache: 0 blocks used  │
└──────────────────────────┘
```

### Memory Efficiency

**Traditional (Contiguous Allocation)**:
- Pre-allocate: 108 slots (entire max_model_len)
- Wasted if request generates <108 tokens
- Memory: 108 slots × model_dim × num_layers × 2 bytes

**PagedAttention (Paged Allocation)**:
- Allocate: 2 blocks = 32 slots (only what's needed)
- Wasted: 24 slots (in partially filled blocks)
- Memory: 32 slots × model_dim × num_layers × 2 bytes
- **Savings**: 70% less memory!

---

## Streaming Response Flow

### Server-Sent Events (SSE) Format

```
# Chunk 1 (first token)
data: {"id":"req-123abc","choices":[{"text":"Quantum","finish_reason":null}]}

# Chunk 2 (second token)
data: {"id":"req-123abc","choices":[{"text":" computing","finish_reason":null}]}

# Chunk 3 (third token)
data: {"id":"req-123abc","choices":[{"text":" uses","finish_reason":null}]}

# ... (97 more chunks)

# Final chunk (no new text, finish_reason set)
data: {"id":"req-123abc","choices":[{"text":"","finish_reason":"length"}]}

# Stream terminator
data: [DONE]
```

### Timing Breakdown

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| **Time to First Token (TTFT)** | 50-500 ms | Prefill latency |
| **Inter-Token Latency (ITL)** | 10-30 ms | Decode latency per token |
| **Total Time (100 tokens)** | 1-3 seconds | TTFT + (100 × ITL) |

**Example**:
- TTFT: 100 ms (process 8-token prompt)
- ITL: 20 ms per token
- Total: 100 ms + (100 × 20 ms) = 2.1 seconds

---

## Error Handling and Edge Cases

### 1. Out of Memory (OOM)

**Scenario**: All KV cache blocks allocated, new request arrives

```python
# Scheduler tries to allocate
new_blocks = kv_cache_manager.allocate_slots(request, num_tokens=8)

if new_blocks is None:
    # No free blocks! Must preempt
    preempted_req = scheduler.running.pop()  # Remove last request
    kv_cache_manager.free(preempted_req)     # Free its blocks
    preempted_req.status = RequestStatus.PREEMPTED
    scheduler.waiting.prepend(preempted_req) # Re-queue at front
```

**User Impact**: Preempted request starts over from beginning

### 2. Request Cancellation

**Scenario**: User disconnects HTTP connection

```python
# API server detects disconnect
abort_request("req-123abc")

# Internally:
output_processor.abort_request("req-123abc")
scheduler.remove_request("req-123abc")
kv_cache_manager.free(request)  # Immediate cleanup
```

### 3. Max Tokens Reached

**Scenario**: Generated 100 tokens, max_tokens=100

```python
if request.num_output_tokens >= request.max_tokens:
    request.finished = True
    finish_reason = "length"
    # Free KV cache on next iteration
```

### 4. Stop String Detected

**Scenario**: User provided `stop=["END"]`, model generates "END"

```python
if "END" in request_state.output_text:
    request.finished = True
    finish_reason = "stop"
    # Trim output to before stop string
    request_state.output_text = request_state.output_text.split("END")[0]
```

---

## Performance Metrics

### Key Metrics Tracked

**Per-Request Metrics**:
- `time_in_queue`: Time in WAITING state
- `time_to_first_token`: Prefill latency
- `inter_token_latency`: Decode latency per token
- `total_time`: End-to-end latency
- `num_preemptions`: How many times preempted

**System-Wide Metrics**:
- `num_requests_running`: Current batch size
- `num_requests_waiting`: Queue depth
- `gpu_cache_usage_perc`: KV cache utilization
- `avg_generation_throughput`: Tokens/sec across all requests
- `num_preemptions_total`: Total preemptions

### Monitoring Example

```python
# vLLM exposes Prometheus metrics at /metrics
curl http://localhost:8000/metrics

# Example output:
# vllm:num_requests_running 42
# vllm:gpu_cache_usage_perc 0.75
# vllm:avg_generation_throughput_toks_per_s 5234.5
# vllm:avg_time_to_first_token_seconds 0.123
```

---

## Summary: Complete Request Lifecycle

1. **HTTP POST** → FastAPI server
2. **Tokenize** → Prompt text → Token IDs
3. **Queue** → WAITING state in Scheduler
4. **Allocate** → KV cache blocks from BlockPool
5. **Prefill** → Process all prompt tokens, populate KV cache
6. **Sample** → First output token
7. **Stream** → Send first token to client
8. **Decode Loop** (×100):
   - Allocate more KV cache slots as needed
   - Model forward pass (1 token)
   - Sample next token
   - Stream to client
9. **Finish** → Detect stop condition (max_tokens, EOS, stop string)
10. **Cleanup** → Free KV cache blocks, remove from tracking

**Total Duration**: ~1-3 seconds for 100 tokens

**Memory Footprint**: 2-4 blocks (32-64 slots) × model_size

**GPU Utilization**: 80-95% (continuous batching keeps GPU busy)

---

## Appendix: Key File References

| Component | Source File |
|-----------|-------------|
| API Server | `vllm/entrypoints/openai/api_server.py` |
| Serving Engine | `vllm/entrypoints/openai/serving_engine.py` |
| LLM Engine | `vllm/v1/engine/llm_engine.py` |
| Processor | `vllm/v1/engine/processor.py` |
| Output Processor | `vllm/v1/engine/output_processor.py` |
| Engine Core | `vllm/v1/engine/core.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` |
| Block Table | `vllm/v1/worker/block_table.py` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` |
| Model Runner | `vllm/v1/worker/gpu_model_runner.py` |
| Sampler | `vllm/v1/sample/sampler.py` |

---

This guide provides a complete trace of a single request through vLLM's serving stack. For further details, refer to the annotated source code in `Claude_tutorials/annotated_code/`.
