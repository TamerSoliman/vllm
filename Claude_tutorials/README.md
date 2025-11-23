# vLLM Core Architecture & Deployment Deep Dive

**A comprehensive tutorial collection for understanding and deploying vLLM in production**

Created: 2025-11-23
Author: Claude (Anthropic AI)
vLLM Version: v0.6.0+

---

## 📚 Overview

This tutorial collection provides an in-depth exploration of vLLM's core architecture, deployment patterns, and performance optimization strategies. Unlike standard documentation, these materials go **deep into the code**, explaining the **What, How, and Why** of every major component.

**Who is this for?**
- **ML Engineers** deploying vLLM in production on AWS EC2 or bare metal
- **System Architects** optimizing high-throughput LLM serving
- **Contributors** wanting to understand vLLM internals
- **Researchers** exploring PagedAttention and continuous batching

---

## 📁 Repository Structure

```
Claude_tutorials/
├── README.md (this file)
├── annotated_code/
│   ├── LLMEngine_annotated.py
│   ├── Scheduler_annotated.py
│   └── BlockTable_PagedAttention_annotated.py
└── guides/
    ├── vLLM_Deployment_Patterns_on_EC2.md
    ├── Configuration_Reference.md
    └── Inference_Request_Lifecycle_Guide.md
```

---

## 🎯 Quick Start

### For Deployment Engineers
**Start here**: [vLLM Deployment Patterns on EC2](guides/vLLM_Deployment_Patterns_on_EC2.md)
- Docker vs bare-metal deployment
- Configuration templates for different workloads
- Production hardening checklist

### For Performance Tuning
**Start here**: [Configuration Reference](guides/Configuration_Reference.md)
- 50+ critical parameters explained
- VRAM and throughput impact analysis
- Memory estimation formulas

### For Architecture Understanding
**Start here**: [Inference Request Lifecycle Guide](guides/Inference_Request_Lifecycle_Guide.md)
- Complete trace of a single request
- Component interaction diagrams
- Memory flow visualization

### For Code Contributors
**Start here**: Annotated source code in `annotated_code/`
- Heavily commented core components
- Design pattern explanations
- Performance optimization insights

---

## 📖 Tutorial Catalog

### 1. Annotated Source Code

These are **line-by-line annotated** versions of vLLM's core components, explaining the architecture decisions and implementation details.

#### 📄 [LLMEngine_annotated.py](annotated_code/LLMEngine_annotated.py)
**Source**: `vllm/v1/engine/llm_engine.py`

**What you'll learn**:
- How LLMEngine orchestrates the entire serving pipeline
- Request intake flow (HTTP → tokenization → queueing)
- Data parallelism coordination
- LoRA multi-tenancy architecture
- Metrics and monitoring hooks

**Key sections**:
- Request processing lifecycle
- Multi-process vs single-process modes
- Data parallel synchronization (dummy batches)
- LoRA adapter management

**Best for**: Understanding the high-level serving architecture

---

#### 📄 [Scheduler_annotated.py](annotated_code/Scheduler_annotated.py)
**Source**: `vllm/v1/core/sched/scheduler.py`

**What you'll learn**:
- **Continuous batching algorithm** (vLLM's core innovation)
- How requests are selected for each iteration
- KV cache allocation and preemption logic
- Priority scheduling and fairness
- Chunked prefill for long contexts

**Key sections**:
- Phase 1: Scheduling RUNNING requests
- Phase 2: Admitting WAITING requests
- Preemption mechanism (when out of memory)
- Prefix caching integration
- LoRA adapter constraint checking

**Best for**: Understanding vLLM's batching magic and throughput optimization

**Critical insights**:
> "There's no explicit 'prefill phase' or 'decode phase'. Each request just has `num_computed_tokens` and `num_tokens_with_spec`. The scheduler tries to close this gap each iteration. This generalizes to chunked prefills, prefix caching, speculative decoding, and regular decode." — From annotations

---

#### 📄 [BlockTable_PagedAttention_annotated.py](annotated_code/BlockTable_PagedAttention_annotated.py)
**Source**: `vllm/v1/worker/block_table.py`

**What you'll learn**:
- **PagedAttention**: How vLLM applies OS paging to KV cache
- Logical-to-physical memory mapping
- Block table structure and slot mapping calculation
- Hybrid block sizes (memory vs kernel granularity)
- Decode Context Parallelism (DCP) support

**Key sections**:
- Virtual memory paging analogy
- `compute_slot_mapping()`: Address translation formula
- KV cache block allocation lifecycle
- 3-4× memory savings vs traditional approach

**Best for**: Understanding PagedAttention's memory efficiency

**Visual example from annotations**:
```
Logical Tokens:  [0, 1, ..., 15] [16, ..., 31] [32, ..., 47]
Logical Blocks:  [   Block 0   ] [  Block 1  ] [  Block 2  ]
Physical Blocks: [     42      ] [     17     ] [     88    ]
Memory Offsets:  [   42 × 16   ] [  17 × 16   ] [  88 × 16  ]

Non-contiguous physical blocks (42, 17, 88) but logically contiguous!
```

---

### 2. Deployment & Configuration Guides

#### 📘 [vLLM Deployment Patterns on EC2](guides/vLLM_Deployment_Patterns_on_EC2.md)

**Complete guide to deploying vLLM on AWS EC2 without EKS or SageMaker**

**Topics covered**:
1. **Hardware Prerequisites**
   - Recommended EC2 instance types (p4d, p4de, g5)
   - VRAM requirements per model size
   - CUDA version compatibility

2. **Method 1: Docker Deployment**
   - Step-by-step Docker setup
   - NVIDIA Container Toolkit installation
   - Production Docker Compose configuration
   - Health checks and auto-restart

3. **Method 2: Bare Metal Deployment**
   - Python virtual environment setup
   - PyTorch + CUDA installation
   - vLLM pip installation
   - systemd service configuration

4. **Configuration Examples**
   - Low latency chat (single GPU)
   - High throughput batch (single GPU)
   - Long context RAG (16K tokens)
   - Multi-GPU tensor parallelism (4× A100)
   - LoRA multi-tenancy

5. **Production Considerations**
   - API key authentication
   - TLS/SSL setup
   - Load balancer configuration (NGINX)
   - Spot instance cost optimization
   - Auto-scaling patterns

6. **Monitoring & Troubleshooting**
   - Key Prometheus metrics
   - OOM debugging
   - Latency optimization
   - Throughput tuning

**Quick start commands**:

Docker (5 minutes):
```bash
docker run -d --gpus all --shm-size 8g -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf
```

Bare metal (30 minutes):
```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm
vllm serve meta-llama/Llama-2-7b-chat-hf
```

**Best for**: DevOps engineers deploying vLLM for the first time

---

#### 📊 [Configuration Reference](guides/Configuration_Reference.md)

**Comprehensive reference for 50+ critical vLLM parameters**

**Structure**:
1. **Model Configuration** (9 parameters)
   - `--model`, `--dtype`, `--quantization`, `--max-model-len`

2. **Parallelism & Distribution** (4 parameters)
   - `--tensor-parallel-size`, `--pipeline-parallel-size`

3. **Memory Management** (5 parameters)
   - `--gpu-memory-utilization` ⭐ Most critical!
   - `--block-size`, `--swap-space`, `--enable-prefix-caching`

4. **Scheduling & Batching** (6 parameters)
   - `--max-num-seqs`, `--max-num-batched-tokens`
   - `--enable-chunked-prefill`

5. **Performance Tuning** (7 parameters)
   - `--enable-cuda-graph`, `--compilation-config`

6. **Server & API Settings** (10 parameters)
   - `--host`, `--port`, `--api-key`, `--ssl-keyfile`

7. **Advanced Features** (10+ parameters)
   - LoRA, speculative decoding, guided decoding

**For each parameter**:
- ✅ **Default value**
- 📝 **Purpose** (what it controls)
- 💾 **VRAM impact** (⭐ to ⭐⭐⭐⭐⭐)
- 🚀 **Throughput impact** (⭐ to ⭐⭐⭐⭐⭐)
- 🎯 **Typical values** (for different use cases)
- 📐 **Memory formulas** (for estimation)

**Example entry**:
```markdown
### `--gpu-memory-utilization`
- **Default**: `0.90`
- **Purpose**: Fraction of GPU VRAM to use for KV cache
- **VRAM Impact**: ⭐⭐⭐⭐⭐ (Directly controls KV cache size)
- **Throughput Impact**: ⭐⭐⭐⭐⭐ (More cache → more requests)
- **Typical Values**:
  - Development: `0.85` (safe margin)
  - Production: `0.90` - `0.95`
  - Aggressive: `0.98` (risk of OOM)
```

**Bonus content**:
- **Configuration Templates**: 5 pre-tuned configs for common workloads
- **Memory Estimation Formulas**: Calculate VRAM requirements
- **Impact Matrix**: Quick reference for parameter tuning priorities
- **Troubleshooting Guide**: Common problems and solutions

**Best for**: Performance engineers tuning vLLM for specific SLOs

---

#### 🔄 [Inference Request Lifecycle Guide](guides/Inference_Request_Lifecycle_Guide.md)

**Complete data flow trace of a single streaming request**

**Follows a request through**:
1. **Phase 1: Request Intake** (HTTP → Internal Request)
   - FastAPI request parsing
   - Tokenization (text → token IDs)
   - Request state creation
   - Queueing to Scheduler

2. **Phase 2: Waiting in Queue**
   - Scheduler queue management
   - Resource availability checking

3. **Phase 3: Prefill** (First Token)
   - KV cache block allocation
   - Input batch preparation
   - Slot mapping calculation (logical → physical)
   - Model forward pass (entire prompt)
   - Token sampling
   - First token streaming

4. **Phase 4: Decode Loop** (Subsequent Tokens)
   - Incremental KV cache growth
   - Single-token forward passes
   - Continuous batching integration
   - Streaming response chunks

5. **Phase 5: Completion & Cleanup**
   - Stop condition detection
   - KV cache deallocation
   - Request state cleanup

**Detailed code examples** for each step:
```python
# Example: Slot mapping calculation
req_indices = [0, 0, 0, 0, 0, 0, 0, 0]  # All from request 0
positions = [0, 1, 2, 3, 4, 5, 6, 7]    # Token positions
block_table[0] = [42]  # Request 0 uses physical block 42

# For each token:
# block_idx = position // block_size
# physical_block = block_table[req_idx][block_idx]
# offset = position % block_size
# slot = physical_block × block_size + offset

slot_mapping = [672, 673, 674, 675, 676, 677, 678, 679]
```

**Visual diagrams**:
- Component interaction flowchart
- KV cache memory state evolution
- Timeline of memory allocation

**Performance breakdown**:
- Time to First Token (TTFT): 50-500 ms
- Inter-Token Latency (ITL): 10-30 ms
- Total time (100 tokens): 1-3 seconds

**Best for**: Understanding vLLM's internal request processing

---

## 🔬 Core Concepts Explained

### PagedAttention

**The Innovation**: Apply OS virtual memory paging to LLM KV cache

**Problem**:
- Traditional: Pre-allocate max_seq_len slots per request
- Waste: Short requests waste memory (e.g., use 50 slots, allocate 2048)

**Solution**:
- PagedAttention: Allocate fixed-size blocks (e.g., 16 tokens) on-demand
- Non-contiguous physical storage, contiguous logical view
- **Result**: 3-4× memory savings, higher batch sizes

**Key files**:
- `BlockTable_PagedAttention_annotated.py`: Address translation
- `vllm/v1/core/kv_cache_manager.py`: Block allocation
- `vllm/attention/ops/paged_attn.py`: CUDA kernel

**Memory formula**:
```
Traditional: max_model_len × num_requests × KV_size
PagedAttention: actual_tokens × KV_size (rounded to blocks)
```

---

### Continuous Batching

**The Innovation**: Dynamically add/remove requests at EVERY iteration

**Problem**:
- Static batching: Wait for N requests → process all N in lockstep
- Issue: Short completions waste GPU waiting for long ones

**Solution**:
- Continuous batching: Add new requests as soon as others finish
- Mixed batching: Prefill and decode in same batch
- **Result**: 10-20× higher throughput, near-100% GPU utilization

**Example timeline**:
```
Iteration 1: [Request A (prefill), Request B (prefill)]
Iteration 2: [Request A (decode), Request B (decode), Request C (prefill)]
Iteration 3: [Request A DONE, Request B (decode), Request C (decode), Request D (prefill)]
```

**Key files**:
- `Scheduler_annotated.py`: Batching algorithm
- `vllm/v1/core/sched/scheduler.py`: Implementation

---

### Tensor Parallelism

**The Purpose**: Split model across multiple GPUs **within a node**

**How it works**:
- Each layer's weights divided across N GPUs
- Example: Attention has 32 heads → TP=4 → 8 heads per GPU
- AllReduce synchronization after each layer

**When to use**:
- Model too large for single GPU (e.g., 70B model)
- TP=2 for 13B models, TP=4/8 for 70B models

**Trade-off**:
- ✅ Enables large model serving
- ❌ 10-20% communication overhead

**Configuration**:
```bash
vllm serve meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 8  # Requires 8× GPUs
```

---

## 🎓 Learning Path

### Beginner: Deploy Your First vLLM Server
1. Read [vLLM Deployment Patterns on EC2](guides/vLLM_Deployment_Patterns_on_EC2.md)
2. Try Docker quick start (5 min)
3. Test with example request
4. Explore [Configuration Reference](guides/Configuration_Reference.md) for your use case

### Intermediate: Optimize for Your Workload
1. Review [Configuration Reference](guides/Configuration_Reference.md) templates
2. Calculate VRAM requirements using formulas
3. Tune `--gpu-memory-utilization`, `--max-num-seqs`, `--max-num-batched-tokens`
4. Monitor Prometheus metrics
5. Iterate based on latency/throughput SLOs

### Advanced: Understand the Architecture
1. Read [Inference Request Lifecycle Guide](guides/Inference_Request_Lifecycle_Guide.md)
2. Study `Scheduler_annotated.py` for batching logic
3. Study `BlockTable_PagedAttention_annotated.py` for memory management
4. Study `LLMEngine_annotated.py` for orchestration
5. Dive into source code with annotations as reference

### Expert: Contribute to vLLM
1. Master all annotated code
2. Understand trade-offs in scheduler decisions
3. Profile with PyTorch profiler (`engine.start_profile()`)
4. Benchmark with vLLM's benchmarking tools
5. Submit optimizations or features

---

## 🛠️ Common Use Cases

### 1. Real-Time Chat (Low Latency)
**Goal**: <100ms TTFT, <20ms ITL

**Config**:
```bash
--max-num-batched-tokens 4096  # Small batches
--max-num-seqs 64              # Few concurrent
--gpu-memory-utilization 0.85  # Headroom for spikes
```

**Read**: Configuration Reference → Template 1

---

### 2. Batch Processing (High Throughput)
**Goal**: >5000 tokens/sec aggregate

**Config**:
```bash
--max-num-batched-tokens 32768  # Large batches
--max-num-seqs 512              # Many concurrent
--enable-prefix-caching         # Cache prompts
```

**Read**: Configuration Reference → Template 2

---

### 3. RAG with Long Context
**Goal**: 16K context, cache documents

**Config**:
```bash
--max-model-len 16384           # Long context
--enable-prefix-caching         # Cache docs
--enable-chunked-prefill        # Chunk long prompts
```

**Read**: Configuration Reference → Template 3

---

### 4. Multi-Tenant LoRA Serving
**Goal**: Serve 8 different fine-tuned adapters

**Config**:
```bash
--enable-lora
--max-loras 8
--max-lora-rank 64
```

**Read**: Configuration Reference → Template 5

---

## 📊 Performance Benchmarks

**Hardware**: 1× NVIDIA A100 (40GB)
**Model**: Llama-2-7B
**Config**: `--gpu-memory-utilization 0.90`

| Metric | Value | Notes |
|--------|-------|-------|
| **Max Concurrent Requests** | 256 | With `--max-num-seqs 256` |
| **Throughput (Decode)** | ~5,000 tokens/sec | Aggregate across all requests |
| **Throughput (Prefill)** | ~15,000 tokens/sec | Parallel processing |
| **TTFT (8-token prompt)** | ~50 ms | Single request |
| **TTFT (512-token prompt)** | ~300 ms | Single request |
| **Inter-Token Latency** | ~15 ms | Batch size = 64 |
| **KV Cache Efficiency** | 75% utilization | vs 20% traditional |
| **Memory Savings** | 3-4× | PagedAttention vs traditional |

**Source**: Official vLLM benchmarks

---

## 🔗 Additional Resources

### Official Documentation
- vLLM Docs: https://docs.vllm.ai
- vLLM GitHub: https://github.com/vllm-project/vllm
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP '23)

### Related Source Files
All paths relative to vLLM repository root:

**Engine**:
- `vllm/v1/engine/llm_engine.py` → See `LLMEngine_annotated.py`
- `vllm/v1/engine/core.py` → EngineCore (main loop)
- `vllm/v1/engine/processor.py` → Input processing
- `vllm/v1/engine/output_processor.py` → Output formatting

**Scheduler**:
- `vllm/v1/core/sched/scheduler.py` → See `Scheduler_annotated.py`
- `vllm/v1/core/kv_cache_manager.py` → KV cache management
- `vllm/v1/core/block_pool.py` → Physical block allocation

**Attention**:
- `vllm/v1/worker/block_table.py` → See `BlockTable_PagedAttention_annotated.py`
- `vllm/attention/ops/paged_attn.py` → CUDA kernels
- `vllm/attention/layer.py` → Attention layer wrapper

**API Server**:
- `vllm/entrypoints/openai/api_server.py` → FastAPI server
- `vllm/entrypoints/openai/serving_engine.py` → Request handling
- `vllm/entrypoints/cli/serve.py` → CLI entry point

**Configuration**:
- `vllm/config/model.py` → ModelConfig
- `vllm/config/scheduler.py` → SchedulerConfig
- `vllm/config/cache.py` → CacheConfig
- `vllm/entrypoints/openai/cli_args.py` → CLI arguments

---

## 🤝 Contributing

Found an error or want to improve these tutorials? Contributions welcome!

**How to contribute**:
1. Fork the vLLM repository
2. Create a branch: `git checkout -b improve-tutorials`
3. Make changes in `Claude_tutorials/`
4. Submit PR to main vLLM repository

**Contribution ideas**:
- Add more configuration examples
- Create benchmarking guides
- Annotate additional source files
- Add troubleshooting case studies
- Translate to other languages

---

## 📝 Changelog

### 2025-11-23: Initial Release
- ✅ Created annotated source code for LLMEngine, Scheduler, BlockTable
- ✅ Wrote comprehensive EC2 deployment guide (Docker + bare metal)
- ✅ Documented 50+ configuration parameters with impact analysis
- ✅ Created complete request lifecycle trace
- ✅ Added configuration templates for common use cases

---

## 📜 License

These tutorials are part of the vLLM project and follow the same Apache 2.0 license.

---

## 🙏 Acknowledgments

Created by **Claude** (Anthropic AI) through deep analysis of the vLLM codebase.

**vLLM Team**: For building an incredible LLM serving system
**vLLM Community**: For extensive documentation and examples
**Paper Authors**: Woosuk Kwon et al. for PagedAttention innovation

---

## 💬 Feedback

Have questions or suggestions? Open an issue in the vLLM repository with `[Tutorial]` prefix.

---

**Happy Serving! 🚀**
