# vLLM Configuration Reference: 50+ Critical Parameters

## Table of Contents
1. [Model Configuration](#1-model-configuration)
2. [Parallelism & Distribution](#2-parallelism--distribution)
3. [Memory Management](#3-memory-management)
4. [Scheduling & Batching](#4-scheduling--batching)
5. [Performance Tuning](#5-performance-tuning)
6. [Server & API Settings](#6-server--api-settings)
7. [Advanced Features](#7-advanced-features)

---

## How to Use This Reference

Each parameter is documented with:
- **Default Value**: What vLLM uses if unspecified
- **Purpose**: What this parameter controls
- **VRAM Impact**: How it affects GPU memory usage
- **Throughput Impact**: Effect on serving throughput
- **Typical Values**: Recommended settings for different scenarios

**Command Line Usage:**
```bash
vllm serve <model_name> --parameter-name value
```

**Python API Usage:**
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="<model_name>",
    parameter_name=value,
)
```

---

## 1. Model Configuration

### 1.1 `--model` (Required)
- **Default**: None (must specify)
- **Purpose**: HuggingFace model ID or local path
- **Examples**:
  - `meta-llama/Llama-2-7b-chat-hf`
  - `/data/models/custom-model`
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Determines model size)
- **Throughput Impact**: Larger models → lower throughput

### 1.2 `--tokenizer`
- **Default**: Same as `--model`
- **Purpose**: Override tokenizer (if different from model)
- **Use Case**: Custom tokenizer for fine-tuned models
- **VRAM Impact**: None
- **Throughput Impact**: None

### 1.3 `--revision`
- **Default**: `"main"`
- **Purpose**: Git branch/tag/commit of HF model
- **Examples**: `"v1.0"`, `"4f2c1e3"`
- **VRAM Impact**: None
- **Throughput Impact**: None

### 1.4 `--tokenizer-mode`
- **Default**: `"auto"`
- **Options**: `"auto"`, `"slow"`, `"mistral"`
- **Purpose**: Tokenizer implementation
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Slow tokenizer adds latency)

### 1.5 `--trust-remote-code`
- **Default**: `False`
- **Purpose**: Allow custom model code from HuggingFace
- **Required For**: Qwen, DeepSeek, many research models
- **VRAM Impact**: None
- **Throughput Impact**: None
- **Security Warning**: ⚠️ Only enable for trusted models!

### 1.6 `--max-model-len`
- **Default**: Model's `max_position_embeddings` (usually 2048-4096)
- **Purpose**: Maximum sequence length (prompt + output)
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Linear with sequence length)
- **Throughput Impact**: ⭐⭐⭐ (Longer sequences → fewer requests fit)
- **Typical Values**:
  - Chatbots: `4096` - `8192`
  - RAG: `16384` - `32768`
  - Code: `8192` - `16384`
- **Memory Formula**: `VRAM_kv_cache ≈ max_model_len × num_seqs × 2 × num_layers × hidden_dim × 2 bytes`

### 1.7 `--dtype`
- **Default**: `"auto"` (uses model's default, usually `float16`)
- **Options**: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`
- **Purpose**: Model weight precision
- **VRAM Impact**:
  - `float32`: 2× memory vs `float16`
  - `bfloat16`: Same as `float16` but better numerical stability
- **Throughput Impact**: ⭐ (`bfloat16` slightly faster on Ampere+)
- **Recommendation**: Use `bfloat16` on A100/H100, `float16` on older GPUs

### 1.8 `--quantization`
- **Default**: `None`
- **Options**: `"awq"`, `"gptq"`, `"squeezellm"`, `"bitsandbytes"`
- **Purpose**: Load quantized models (4-bit, 8-bit)
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (50-75% reduction!)
- **Throughput Impact**: ⭐⭐ (Slight dequantization overhead)
- **Example**: `--quantization awq --model TheBloke/Llama-2-7B-AWQ`

### 1.9 `--load-format`
- **Default**: `"auto"`
- **Options**: `"auto"`, `"pt"`, `"safetensors"`, `"npcache"`, `"dummy"`
- **Purpose**: Model weight file format
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Affects load time only)
- **Recommendation**: `safetensors` for faster/safer loading

---

## 2. Parallelism & Distribution

### 2.1 `--tensor-parallel-size` (TP)
- **Default**: `1`
- **Purpose**: Split model across N GPUs **within a node**
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Divides model memory by TP)
- **Throughput Impact**: ⭐⭐ (Communication overhead ~10-20%)
- **Typical Values**:
  - 7B model: `1` (single GPU)
  - 13B model: `2` (2 GPUs)
  - 70B model: `4` or `8`
- **Constraints**:
  - Must divide number of attention heads
  - TP × PP ≤ total GPUs
- **Example**: 4× A100 (40GB each) → `--tensor-parallel-size 4`

### 2.2 `--pipeline-parallel-size` (PP)
- **Default**: `1`
- **Purpose**: Split model **layers** across GPUs
- **VRAM Impact**: ⭐⭐⭐⭐ (Divides layer memory)
- **Throughput Impact**: ⭐⭐⭐ (Pipeline bubbles reduce utilization)
- **Use Case**: Very large models (70B+) with many GPUs
- **Recommendation**: Prefer TP over PP when possible
- **Example**: 8× GPUs → `--tensor-parallel-size 4 --pipeline-parallel-size 2`

### 2.3 `--distributed-executor-backend`
- **Default**: `"ray"`
- **Options**: `"ray"`, `"mp"` (multiprocessing)
- **Purpose**: Multi-GPU coordination backend
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Ray has slight overhead)
- **Recommendation**: Use `ray` for multi-node, `mp` for single-node

### 2.4 `--worker-use-ray`
- **Default**: `False`
- **Purpose**: Use Ray for worker processes (deprecated, use `--distributed-executor-backend`)
- **VRAM Impact**: None
- **Throughput Impact**: ⭐

---

## 3. Memory Management

### 3.1 `--gpu-memory-utilization`
- **Default**: `0.90`
- **Purpose**: Fraction of GPU VRAM to use for KV cache
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Directly controls KV cache size)
- **Throughput Impact**: ⭐⭐⭐⭐⭐ (More cache → more requests)
- **Typical Values**:
  - Development: `0.85` (safe margin)
  - Production: `0.90` - `0.95`
  - Aggressive: `0.98` (risk of OOM)
- **Formula**: `kv_cache_blocks = (total_vram - model_size) × gpu_memory_utilization / block_size`
- **⚠️ Warning**: Values >0.95 may cause OOM under load spikes

### 3.2 `--block-size`
- **Default**: `16` (determined by attention backend)
- **Purpose**: KV cache block size (tokens per block)
- **VRAM Impact**: ⭐ (Affects memory fragmentation)
- **Throughput Impact**: ⭐ (Larger blocks → less overhead)
- **Typical Values**: `16`, `32` (backend-dependent)
- **Recommendation**: Leave at default unless using custom kernels

### 3.3 `--swap-space`
- **Default**: `4` (GB)
- **Purpose**: CPU RAM for swapping preempted requests
- **VRAM Impact**: None (uses CPU RAM)
- **Throughput Impact**: ⭐⭐ (Enables request preemption)
- **Use Case**: Bursty traffic with priority queues
- **Recommendation**: `4-8 GB` for production

### 3.4 `--cpu-offload-gb`
- **Default**: `0`
- **Purpose**: Offload model weights to CPU RAM
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Reduces GPU memory)
- **Throughput Impact**: ⭐⭐⭐⭐ (Slow! PCIe overhead)
- **Use Case**: Run models too large for GPU VRAM
- **Recommendation**: Only for development/testing

### 3.5 `--enable-prefix-caching`
- **Default**: `False`
- **Purpose**: Cache KV for common prompt prefixes (Automatic Prefix Caching)
- **VRAM Impact**: None (uses existing KV cache)
- **Throughput Impact**: ⭐⭐⭐⭐⭐ (10-100× speedup for RAG!)
- **Use Case**: RAG, chatbots with system prompts, few-shot prompts
- **Recommendation**: **Always enable** for production!
- **Example Speedup**: Shared 1000-token system prompt → 1000 tokens cached across all requests

---

## 4. Scheduling & Batching

### 4.1 `--max-num-seqs`
- **Default**: `256`
- **Purpose**: Maximum concurrent requests in running queue
- **VRAM Impact**: ⭐⭐⭐ (More seqs → more KV cache)
- **Throughput Impact**: ⭐⭐⭐⭐⭐ (Higher = more throughput)
- **Typical Values**:
  - Low latency: `32` - `64`
  - Balanced: `128` - `256`
  - High throughput: `512` - `1024`
- **Constraint**: Limited by KV cache capacity
- **Formula**: `max_num_seqs ≈ kv_cache_blocks / (max_model_len / block_size)`

### 4.2 `--max-num-batched-tokens`
- **Default**: `max_model_len × max_num_seqs` (no limit)
- **Purpose**: Maximum tokens per forward pass (controls batch size)
- **VRAM Impact**: ⭐⭐ (Compute memory, not KV cache)
- **Throughput Impact**: ⭐⭐⭐⭐⭐ (Latency-throughput tradeoff)
- **Typical Values**:
  - Low latency: `2048` - `4096` (small batches, fast iteration)
  - High throughput: `16384` - `32768` (large batches, max GPU util)
- **Recommendation**:
  - **Latency-critical**: Set to `2× max_model_len`
  - **Throughput-critical**: Leave unlimited or set to `8× max_model_len`
- **Example**: For `max_model_len=4096`, use `--max-num-batched-tokens 8192` for low latency

### 4.3 `--max-paddings`
- **Default**: `256`
- **Purpose**: Maximum padding tokens in batch
- **VRAM Impact**: ⭐
- **Throughput Impact**: ⭐⭐ (Padding wastes compute)
- **Recommendation**: Leave at default

### 4.4 `--scheduling-policy`
- **Default**: `"fcfs"` (First-Come-First-Served)
- **Options**: `"fcfs"`, `"priority"`
- **Purpose**: Request scheduling algorithm
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Priority enables SLO optimization)
- **Use Case**: `priority` for multi-tenant or SLA-bound serving

### 4.5 `--enable-chunked-prefill`
- **Default**: `False`
- **Purpose**: Split long prompts into chunks (mixed prefill/decode batching)
- **VRAM Impact**: None
- **Throughput Impact**: ⭐⭐⭐⭐ (Reduces TTFT variance)
- **Use Case**: Long document processing, RAG with large contexts
- **Benefit**: Prevents single long prompt from blocking short requests

### 4.6 `--max-num-on-the-fly-batching`
- **Default**: `256`
- **Purpose**: Max new requests added to batch between iterations
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Affects scheduler overhead)
- **Recommendation**: Leave at default

---

## 5. Performance Tuning

### 5.1 `--disable-log-stats`
- **Default**: `False`
- **Purpose**: Disable periodic performance logging
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Logging has minimal overhead)
- **Recommendation**: Enable logging in production for monitoring

### 5.2 `--disable-log-requests`
- **Default**: `False`
- **Purpose**: Disable per-request logging
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Reduces log volume)
- **Recommendation**: Disable in high-QPS production

### 5.3 `--enforce-eager`
- **Default**: `False`
- **Purpose**: Disable CUDA graph optimization
- **VRAM Impact**: ⭐⭐ (CUDA graphs use extra memory)
- **Throughput Impact**: ⭐⭐⭐ (Graphs are 10-15% faster)
- **Use Case**: Enable when debugging, disable for production
- **Recommendation**: Leave `False` (use CUDA graphs) for best performance

### 5.4 `--enable-cuda-graph`
- **Default**: `True`
- **Purpose**: Use CUDA graph optimization for decode
- **VRAM Impact**: ⭐⭐ (Allocates graph memory)
- **Throughput Impact**: ⭐⭐⭐ (10-15% speedup)
- **Recommendation**: Always keep enabled
- **Incompatible With**: Some custom ops, certain quantization schemes

### 5.5 `--max-context-len-to-capture`
- **Default**: `8192`
- **Purpose**: Maximum sequence length for CUDA graph capture
- **VRAM Impact**: ⭐⭐ (Larger graphs use more memory)
- **Throughput Impact**: ⭐⭐
- **Recommendation**: Match to typical sequence lengths

### 5.6 `--compilation-config`
- **Default**: `None`
- **Purpose**: PyTorch 2.0 compilation (torch.compile)
- **VRAM Impact**: ⭐
- **Throughput Impact**: ⭐⭐⭐⭐ (Can be 2× faster)
- **Status**: Experimental
- **Example**: `--compilation-config '{"level": 2}'`

### 5.7 `--num-scheduler-steps`
- **Default**: `1`
- **Purpose**: Number of scheduling steps per model forward pass
- **VRAM Impact**: None
- **Throughput Impact**: ⭐⭐ (Higher = more scheduling overhead)
- **Recommendation**: Leave at `1` unless using speculative decoding

---

## 6. Server & API Settings

### 6.1 `--host`
- **Default**: `"localhost"`
- **Purpose**: Server bind address
- **Typical Values**:
  - Local dev: `"localhost"` or `"127.0.0.1"`
  - Production: `"0.0.0.0"` (all interfaces)
- **VRAM Impact**: None
- **Throughput Impact**: None

### 6.2 `--port`
- **Default**: `8000`
- **Purpose**: Server port
- **VRAM Impact**: None
- **Throughput Impact**: None

### 6.3 `--api-key`
- **Default**: `None`
- **Purpose**: Require Bearer token authentication
- **Example**: `--api-key "sk-1234567890abcdef"`
- **VRAM Impact**: None
- **Throughput Impact**: None
- **Recommendation**: **Always set** in production!

### 6.4 `--ssl-keyfile` / `--ssl-certfile`
- **Default**: `None`
- **Purpose**: Enable HTTPS with TLS/SSL
- **Example**:
  ```bash
  --ssl-keyfile /path/to/key.pem \
  --ssl-certfile /path/to/cert.pem
  ```
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (SSL overhead ~1-2%)

### 6.5 `--root-path`
- **Default**: `None`
- **Purpose**: Base path for API (for reverse proxy)
- **Example**: `--root-path "/v1/llm"` → API at `/v1/llm/v1/completions`
- **VRAM Impact**: None
- **Throughput Impact**: None

### 6.6 `--served-model-name`
- **Default**: Same as `--model`
- **Purpose**: Model name returned in API responses
- **Example**: `--served-model-name "my-custom-7b"`
- **VRAM Impact**: None
- **Throughput Impact**: None
- **Use Case**: Alias long HuggingFace model IDs

### 6.7 `--chat-template`
- **Default**: Model's default (from tokenizer_config.json)
- **Purpose**: Override Jinja2 chat template
- **Example**: `--chat-template /path/to/template.jinja`
- **VRAM Impact**: None
- **Throughput Impact**: None
- **Use Case**: Custom chat formats for fine-tuned models

### 6.8 `--response-role`
- **Default**: `"assistant"`
- **Purpose**: Role name for model responses in chat
- **VRAM Impact**: None
- **Throughput Impact**: None

### 6.9 `--disable-log-requests`
- **Default**: `False`
- **Purpose**: Don't log every API request (reduces log spam)
- **VRAM Impact**: None
- **Throughput Impact**: ⭐ (Minimal)
- **Recommendation**: Enable for high-QPS production

### 6.10 `--max-log-len`
- **Default**: `None` (unlimited)
- **Purpose**: Truncate prompts/outputs in logs
- **Example**: `--max-log-len 100` (log first 100 chars)
- **VRAM Impact**: None
- **Throughput Impact**: None
- **Recommendation**: Set to `100-200` to avoid log bloat

---

## 7. Advanced Features

### 7.1 `--enable-lora`
- **Default**: `False`
- **Purpose**: Enable LoRA adapter serving (multi-tenancy)
- **VRAM Impact**: ⭐⭐ (Per-adapter overhead)
- **Throughput Impact**: ⭐⭐ (Adapter switching overhead)
- **Use Case**: Serve multiple fine-tuned adapters on same base model
- **Example**:
  ```bash
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64
  ```

### 7.2 `--max-loras`
- **Default**: `1`
- **Purpose**: Maximum LoRA adapters loaded simultaneously
- **VRAM Impact**: ⭐⭐ (Linear with num adapters)
- **Throughput Impact**: ⭐
- **Typical Values**: `4` - `8`

### 7.3 `--max-lora-rank`
- **Default**: `16`
- **Purpose**: Maximum LoRA rank (dimensionality)
- **VRAM Impact**: ⭐⭐ (Higher rank → more memory)
- **Throughput Impact**: ⭐ (Higher rank → slower)
- **Typical Values**: `8`, `16`, `32`, `64`

### 7.4 `--speculative-model`
- **Default**: `None`
- **Purpose**: Enable speculative decoding with draft model
- **VRAM Impact**: ⭐⭐⭐ (Loads draft model)
- **Throughput Impact**: ⭐⭐⭐⭐ (2-4× speedup for latency)
- **Example**: `--speculative-model "meta-llama/Llama-2-7b-hf"` (for 70B target)
- **Use Case**: Latency-critical serving with large models

### 7.5 `--num-speculative-tokens`
- **Default**: `5`
- **Purpose**: Number of tokens to speculate per step
- **VRAM Impact**: ⭐
- **Throughput Impact**: ⭐⭐⭐
- **Typical Values**: `3` - `7`

### 7.6 `--guided-decoding-backend`
- **Default**: `"outlines"`
- **Options**: `"outlines"`, `"lm-format-enforcer"`
- **Purpose**: Structured output (JSON schemas, regex)
- **VRAM Impact**: ⭐
- **Throughput Impact**: ⭐⭐ (Constrained decoding overhead)
- **Use Case**: Guaranteed JSON output, function calling

### 7.7 `--image-input-type`
- **Default**: `"pixel_values"`
- **Options**: `"pixel_values"`, `"image_features"`
- **Purpose**: Multimodal vision input format
- **VRAM Impact**: ⭐⭐
- **Throughput Impact**: ⭐
- **Use Case**: Vision-language models (LLaVA, Qwen-VL, etc.)

### 7.8 `--image-token-id`
- **Default**: Model-specific
- **Purpose**: Special token ID for image placeholders
- **VRAM Impact**: None
- **Throughput Impact**: None

### 7.9 `--image-input-shape`
- **Default**: Model-specific
- **Purpose**: Input image dimensions
- **Example**: `"1,3,224,224"`
- **VRAM Impact**: ⭐⭐
- **Throughput Impact**: ⭐

### 7.10 `--disable-sliding-window`
- **Default**: `False`
- **Purpose**: Disable sliding window attention (for Mistral, etc.)
- **VRAM Impact**: ⭐⭐⭐ (Sliding window saves memory)
- **Throughput Impact**: ⭐
- **Recommendation**: Keep enabled (don't disable)

---

## Configuration Templates

### Template 1: Low Latency Chat (Single GPU)
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --dtype bfloat16
```
**Expected**: <100ms TTFT, ~50 tokens/sec per request

### Template 2: High Throughput Batch (Single GPU)
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16
```
**Expected**: ~5000 tokens/sec aggregate throughput

### Template 3: Long Context RAG (Single GPU)
```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.98 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16
```
**Expected**: Cache 10K+ token documents, 20-30 concurrent requests

### Template 4: Large Model Multi-GPU (4× A100)
```bash
vllm serve meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --dtype bfloat16
```
**Expected**: 70B model, ~30 tokens/sec per request

### Template 5: Production with LoRA (Multi-Tenant)
```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --enable-lora \
  --max-loras 8 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --api-key "$(openssl rand -hex 32)"
```
**Expected**: Serve 8 different fine-tuned adapters simultaneously

---

## Memory Estimation Formulas

### Model Weights
```
model_memory_gb = num_parameters × bytes_per_param / (1024^3)

bytes_per_param:
- float32: 4
- float16/bfloat16: 2
- int8: 1
- int4 (quantized): 0.5
```

### KV Cache
```
kv_cache_gb = (
    max_num_seqs × max_model_len × 2 (K + V) × num_layers × hidden_dim × 2 (bytes) / (1024^3)
)
```

### Total VRAM Required
```
total_vram_gb = (
    model_memory_gb
    + kv_cache_gb
    + activation_memory_gb (≈ 2-4 GB)
    + overhead_gb (≈ 1-2 GB)
)
```

### Example: Llama-2-7B on A100 (40GB)
```
Model: 7B × 2 bytes = 14 GB
KV Cache: 256 seqs × 4096 tokens × 2 × 32 layers × 4096 hidden_dim × 2 bytes / (1024^3)
        = 256 × 4096 × 2 × 32 × 4096 × 2 / (1024^3)
        ≈ 26 GB

Total: 14 + 26 + 4 = 44 GB → Won't fit! Need to reduce max_num_seqs or max_model_len
```

**Solution**: Reduce `max_num_seqs` to 128:
```
KV Cache: 128 × 4096 × 2 × 32 × 4096 × 2 / (1024^3) ≈ 13 GB
Total: 14 + 13 + 4 = 31 GB → Fits with 9 GB headroom ✓
```

---

## Quick Reference: Impact Matrix

| Parameter | VRAM | Throughput | Latency | When to Tune |
|-----------|------|------------|---------|--------------|
| `--gpu-memory-utilization` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | Always |
| `--max-num-seqs` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Always |
| `--max-model-len` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Always |
| `--max-num-batched-tokens` | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Latency-sensitive |
| `--tensor-parallel-size` | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | Large models |
| `--enable-prefix-caching` | None | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | RAG, chatbots |
| `--enable-chunked-prefill` | None | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Long contexts |
| `--dtype` | ⭐⭐⭐ | ⭐⭐ | ⭐ | Memory-constrained |
| `--quantization` | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | VRAM-constrained |

---

## Source Code References

All parameters are defined in:
- `vllm/config/model.py` - ModelConfig
- `vllm/config/scheduler.py` - SchedulerConfig
- `vllm/config/cache.py` - CacheConfig
- `vllm/config/parallel.py` - ParallelConfig
- `vllm/entrypoints/openai/cli_args.py` - CLI argument parser

---

## Troubleshooting Guide

### Problem: OOM (Out of Memory)
**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions** (in order of impact):
1. ↓ `--gpu-memory-utilization` (0.95 → 0.85)
2. ↓ `--max-model-len` (4096 → 2048)
3. ↓ `--max-num-seqs` (256 → 128)
4. ↑ `--tensor-parallel-size` (1 → 2, requires multi-GPU)
5. Enable `--quantization awq` (requires quantized model)

### Problem: Low Throughput
**Symptoms**: <1000 tokens/sec on single A100

**Solutions**:
1. ↑ `--max-num-seqs` (128 → 512)
2. ↑ `--max-num-batched-tokens` (4096 → 16384)
3. ↑ `--gpu-memory-utilization` (0.85 → 0.95)
4. Enable `--enable-prefix-caching`
5. Use `--dtype bfloat16` (on Ampere+)

### Problem: High Latency (Slow TTFT)
**Symptoms**: >1 second time-to-first-token

**Solutions**:
1. ↓ `--max-num-batched-tokens` (16384 → 4096)
2. Enable `--enable-chunked-prefill`
3. ↓ `--max-num-seqs` (512 → 64)
4. Disable `--disable-log-requests`
5. Use `--speculative-model` (for large models)

---

This reference covers the 50+ most critical configuration parameters. For the complete list, see the official vLLM documentation.
