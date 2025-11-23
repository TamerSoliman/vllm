# vLLM Deployment Patterns on EC2: Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Hardware Prerequisites](#hardware-prerequisites)
3. [Deployment Method 1: Docker Container](#method-1-docker-container-deployment)
4. [Deployment Method 2: Bare Metal Python/pip](#method-2-bare-metal-pythonpip-deployment)
5. [Configuration Examples](#configuration-examples)
6. [Production Considerations](#production-considerations)
7. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)

---

## Overview

vLLM can be deployed on AWS EC2 in two primary ways:
1. **Docker Container**: Pre-built environment with all dependencies
2. **Bare Metal**: Direct Python installation for maximum control

This guide focuses on **bare-metal EC2 deployments** (no EKS, no SageMaker), ideal for:
- Cost-optimized serving on reserved/spot instances
- Full control over environment and dependencies
- Minimal operational overhead
- Direct GPU access with maximum performance

---

## Hardware Prerequisites

### Recommended EC2 Instance Types

#### For Production (Llama-2-7B and similar):
- **p4d.24xlarge**: 8× A100 (40GB), best price/performance
- **p4de.24xlarge**: 8× A100 (80GB), for large models
- **g5.12xlarge**: 4× A10G (24GB), cost-effective alternative

#### For Development/Testing:
- **g5.2xlarge**: 1× A10G (24GB), single GPU testing
- **p3.2xlarge**: 1× V100 (16GB), older but reliable

### Minimum Requirements:
- **GPU**: NVIDIA GPU with Compute Capability ≥ 7.0 (Volta+)
- **VRAM**: ≥ 16GB for 7B models, ≥ 40GB for 13B models
- **System RAM**: ≥ 2× VRAM (for model loading)
- **Disk**: ≥ 100GB SSD (for model weights + Docker image)
- **CUDA**: 12.1+ recommended (vLLM supports 11.8-12.9)

---

## Method 1: Docker Container Deployment

### Advantages
✅ **Zero dependency management**: All CUDA, PyTorch, libraries pre-installed
✅ **Reproducible builds**: Same environment across dev/staging/prod
✅ **Easy updates**: Pull new image to upgrade vLLM version
✅ **Isolation**: Won't conflict with system Python/CUDA

### Disadvantages
❌ **Larger disk footprint**: Image is ~15GB compressed
❌ **Slight overhead**: Docker adds minimal latency (~1%)
❌ **Less flexibility**: Harder to patch individual components

---

### Step-by-Step: Docker Deployment

#### 1. Launch EC2 Instance

```bash
# Example: Launch g5.2xlarge (1× A10G) in us-east-1
aws ec2 run-instances \
  --image-id ami-0c94855ba95c71c99 \  # Ubuntu 22.04 Deep Learning AMI
  --instance-type g5.2xlarge \
  --key-name your-keypair \
  --security-group-ids sg-xxxxx \  # Allow ports 22, 8000
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200}' \
  --iam-instance-profile Name=EC2-S3-Access  # For pulling models from S3
```

SSH into instance:
```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>
```

#### 2. Install NVIDIA Container Toolkit

```bash
# Install Docker (if not pre-installed)
sudo apt-get update
sudo apt-get install -y docker.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access in Docker
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA A10G         Off  | 00000000:00:1E.0 Off |                    0 |
+-------------------------------+----------------------+----------------------+
```

#### 3. Pull vLLM Docker Image

```bash
# Pull latest stable release (recommended for production)
sudo docker pull vllm/vllm-openai:latest

# OR pull specific version for reproducibility
sudo docker pull vllm/vllm-openai:v0.6.0

# Verify image
sudo docker images | grep vllm
```

#### 4. Download Model Weights

Option A: From HuggingFace Hub (requires internet):
```bash
# Model will be cached in /root/.cache/huggingface inside container
# No pre-download needed - vLLM downloads on first run
```

Option B: Pre-download to EC2 disk (for air-gapped or faster startup):
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download model
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir /data/models/llama-2-7b-chat

# Verify download
ls -lh /data/models/llama-2-7b-chat
# Should see: config.json, pytorch_model.bin, tokenizer_model.model, etc.
```

#### 5. Run vLLM Server in Docker

**Basic Example (7B model on single GPU):**

```bash
sudo docker run -d \
  --name vllm-server \
  --gpus all \
  --shm-size 8g \
  -p 8000:8000 \
  -v /data/models:/models \
  -e HF_TOKEN=your_huggingface_token \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 256
```

**Explanation of key flags:**
- `--gpus all`: Expose all GPUs to container
- `--shm-size 8g`: Shared memory for PyTorch DataLoader (prevents OOM)
- `-v /data/models:/models`: Mount model directory (if pre-downloaded)
- `--gpu-memory-utilization 0.90`: Use 90% of VRAM for KV cache (safe default)
- `--max-model-len 4096`: Maximum sequence length (prompt + output)
- `--max-num-seqs 256`: Maximum concurrent requests

**Production Example (Multi-GPU with Tensor Parallelism):**

```bash
# For 4-GPU g5.12xlarge
sudo docker run -d \
  --name vllm-server \
  --gpus all \
  --shm-size 32g \
  -p 8000:8000 \
  --restart unless-stopped \
  -v /data/models:/models \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --enable-prefix-caching \
  --disable-log-requests \
  --trust-remote-code
```

**Advanced flags:**
- `--tensor-parallel-size 4`: Split model across 4 GPUs
- `--max-num-batched-tokens 16384`: Token budget per batch (controls latency/throughput trade-off)
- `--enable-prefix-caching`: Cache common prompt prefixes (huge speedup for similar prompts)
- `--disable-log-requests`: Reduce log verbosity in production
- `--trust-remote-code`: Required for some models (e.g., Qwen, DeepSeek)

#### 6. Verify Server is Running

```bash
# Check container logs
sudo docker logs -f vllm-server

# Expected output:
# INFO:     Started server process [1]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Test API:
```bash
curl http://localhost:8000/v1/models
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-2-7b-chat-hf",
      "object": "model",
      "created": 1234567890,
      "owned_by": "vllm"
    }
  ]
}
```

Test completion:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "San Francisco is a",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

#### 7. Production-Grade Docker Deployment

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  vllm:
    image: vllm/vllm-openai:v0.6.0
    container_name: vllm-server
    restart: unless-stopped
    shm_size: 32gb
    ports:
      - "8000:8000"
    volumes:
      - /data/models:/models
      - /data/logs:/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0,1,2,3
      - HF_HOME=/models/.cache
      - VLLM_LOGGING_LEVEL=INFO
    command: >
      --model meta-llama/Llama-2-13b-chat-hf
      --tensor-parallel-size 4
      --host 0.0.0.0
      --port 8000
      --gpu-memory-utilization 0.95
      --max-model-len 8192
      --max-num-seqs 512
      --enable-prefix-caching
      --served-model-name llama-2-13b-chat
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Start with:
```bash
sudo docker-compose up -d
```

---

## Method 2: Bare Metal Python/pip Deployment

### Advantages
✅ **Maximum performance**: No container overhead
✅ **Full control**: Easy to patch, modify, debug
✅ **Smaller footprint**: Only install what you need
✅ **Faster iterations**: No image rebuild for code changes

### Disadvantages
❌ **Complex setup**: Must manage CUDA, PyTorch, system dependencies
❌ **Dependency hell**: Version conflicts possible
❌ **Environment drift**: Dev vs prod differences

---

### Step-by-Step: Bare Metal Deployment

#### 1. Launch EC2 Instance

Same as Docker method, but use **Deep Learning AMI** (has CUDA pre-installed):

```bash
aws ec2 run-instances \
  --image-id ami-0c94855ba95c71c99 \  # Ubuntu 22.04 Deep Learning AMI
  --instance-type g5.2xlarge \
  --key-name your-keypair \
  --security-group-ids sg-xxxxx \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200}'
```

#### 2. Verify CUDA Installation

```bash
ssh -i your-keypair.pem ubuntu@<instance-ip>

# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Expected: CUDA 12.1 or later
```

If CUDA not installed:
```bash
# Install CUDA 12.1 (example for Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 3. Set Up Python Environment

```bash
# Install Python 3.10+ (if not present)
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv vllm-env
source vllm-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 4. Install PyTorch with CUDA Support

**CRITICAL**: Must install PyTorch with correct CUDA version!

```bash
# For CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
# Expected: True, <number of GPUs>
```

#### 5. Install vLLM

```bash
# Install latest stable release
pip install vllm

# OR install specific version
pip install vllm==0.6.0

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

If you encounter build errors:
```bash
# Install build dependencies
sudo apt-get install -y build-essential cmake ninja-build

# Install vLLM with verbose output to debug
pip install vllm --no-cache-dir -v
```

#### 6. Download Model Weights

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login to HuggingFace (for gated models like Llama)
huggingface-cli login
# Enter your HF token

# Download model
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir /home/ubuntu/models/llama-2-7b-chat \
  --local-dir-use-symlinks False
```

#### 7. Run vLLM Server

**Option A: Using `vllm serve` CLI** (Recommended)

```bash
# Activate venv
source vllm-env/bin/activate

# Start server
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  > /home/ubuntu/logs/vllm.log 2>&1 &

# Check logs
tail -f /home/ubuntu/logs/vllm.log
```

**Option B: Using Python API** (For custom integration)

Create `serve.py`:
```python
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio

async def main():
    engine_args = AsyncEngineArgs(
        model="meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=256,
        enable_prefix_caching=True,
    )
    await run_server(engine_args)

if __name__ == "__main__":
    asyncio.run(main())
```

Run:
```bash
python serve.py > /home/ubuntu/logs/vllm.log 2>&1 &
```

#### 8. Create Systemd Service (Production)

Create `/etc/systemd/system/vllm.service`:
```ini
[Unit]
Description=vLLM OpenAI-Compatible Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
Environment="PATH=/home/ubuntu/vllm-env/bin:/usr/local/cuda-12.1/bin:/usr/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64"
Environment="CUDA_VISIBLE_DEVICES=0,1,2,3"
ExecStart=/home/ubuntu/vllm-env/bin/vllm serve meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --enable-prefix-caching
Restart=always
RestartSec=10
StandardOutput=append:/home/ubuntu/logs/vllm.log
StandardError=append:/home/ubuntu/logs/vllm-error.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

View logs:
```bash
sudo journalctl -u vllm -f
```

---

## Configuration Examples

### Example 1: High Throughput (Batch Processing)

**Use case**: Offline batch inference, data labeling, synthetic data generation

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 2048 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

**Rationale**:
- `--max-num-seqs 512`: Many concurrent requests
- `--max-num-batched-tokens 32768`: Large batches for throughput
- `--enable-chunked-prefill`: Process long prompts in chunks
- Lower `max-model-len`: Shorter sequences = more requests fit

### Example 2: Low Latency (Real-Time Chat)

**Use case**: Chatbots, interactive applications, customer support

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 4096 \
  --disable-log-requests
```

**Rationale**:
- `--max-num-seqs 64`: Fewer concurrent requests (lower latency)
- `--max-num-batched-tokens 4096`: Small batches (fast iteration)
- `--gpu-memory-utilization 0.85`: Leave headroom for spiky traffic

### Example 3: Long Context (RAG, Document Analysis)

**Use case**: Retrieval-augmented generation, document Q&A

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

**Rationale**:
- `--max-model-len 16384`: Long context window
- `--max-num-seqs 32`: Fewer requests (long sequences use more memory)
- `--enable-prefix-caching`: Cache document prefixes (RAG speedup)

### Example 4: Multi-GPU Tensor Parallelism

**Use case**: Large models (70B+) that don't fit on single GPU

```bash
vllm serve meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 128 \
  --trust-remote-code
```

**Rationale**:
- `--tensor-parallel-size 8`: Split model across 8 GPUs
- Must run on instance with 8 GPUs (e.g., p4d.24xlarge)

---

## Production Considerations

### 1. Security Hardening

**API Key Authentication:**
```bash
# Generate API key
export VLLM_API_KEY=$(openssl rand -hex 32)

# Start server with auth
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --api-key $VLLM_API_KEY \
  --host 0.0.0.0 \
  --port 8000
```

Client usage:
```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-2-7b-chat-hf", "prompt": "Hello", "max_tokens": 50}'
```

**Firewall Rules:**
```bash
# Only allow traffic from specific IPs
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw enable
```

**TLS/SSL:**
```bash
# Generate self-signed cert (for testing)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start server with SSL
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --ssl-keyfile key.pem \
  --ssl-certfile cert.pem \
  --host 0.0.0.0 \
  --port 8443
```

### 2. High Availability

**Load Balancer Setup (NGINX):**
```nginx
upstream vllm_backends {
    least_conn;
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://vllm_backends;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

**Health Checks:**
```bash
# vLLM exposes /health endpoint
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

### 3. Cost Optimization

**Use Spot Instances:**
```bash
# Request spot instance (70-90% cheaper)
aws ec2 request-spot-instances \
  --spot-price "2.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification file://spot-spec.json
```

`spot-spec.json`:
```json
{
  "ImageId": "ami-0c94855ba95c71c99",
  "KeyName": "your-keypair",
  "InstanceType": "g5.2xlarge",
  "SecurityGroupIds": ["sg-xxxxx"]
}
```

**Auto-scaling (for variable load):**
```bash
# Use AWS Auto Scaling Groups
# Scale up during business hours, down at night
```

---

## Monitoring and Troubleshooting

### Key Metrics to Monitor

**GPU Utilization:**
```bash
# Install nvidia-smi dashboard
watch -n 1 nvidia-smi
```

**vLLM Prometheus Metrics:**
```bash
# Enable Prometheus endpoint
vllm serve meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-metrics

# Scrape metrics
curl http://localhost:8000/metrics
```

Key metrics:
- `vllm:num_requests_running`: Current batch size
- `vllm:gpu_cache_usage_perc`: KV cache utilization
- `vllm:avg_generation_throughput_toks_per_s`: Throughput
- `vllm:avg_time_to_first_token_seconds`: TTFT latency

### Common Issues

**1. Out of Memory (OOM)**

Symptom:
```
RuntimeError: CUDA out of memory. Tried to allocate 1024.00 MiB
```

Solutions:
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.85  # From 0.90

# Reduce max sequence length
--max-model-len 2048  # From 4096

# Reduce concurrent requests
--max-num-seqs 128  # From 256

# Use tensor parallelism (split across GPUs)
--tensor-parallel-size 2
```

**2. Slow First Token (High TTFT)**

Symptom: Requests take 5-10 seconds before first token appears

Solutions:
```bash
# Enable chunked prefill
--enable-chunked-prefill

# Reduce max batch tokens
--max-num-batched-tokens 4096  # Prioritize latency over throughput

# Use prefix caching
--enable-prefix-caching  # Cache common prompts
```

**3. Low Throughput**

Symptom: Only 10-20 tokens/sec per request

Solutions:
```bash
# Increase batch size
--max-num-batched-tokens 16384  # From 8192
--max-num-seqs 512  # From 256

# Enable continuous batching (on by default)
# Verify not disabled with --disable-log-stats

# Check GPU utilization
nvidia-smi  # Should be 90-100%
```

**4. Model Download Fails**

Symptom:
```
HTTPError: 401 Client Error: Unauthorized
```

Solutions:
```bash
# Login to HuggingFace
huggingface-cli login

# OR set token env var
export HF_TOKEN=your_token_here

# Verify access to gated models (Llama requires approval)
```

---

## Summary: Docker vs Bare Metal

| Factor | Docker | Bare Metal |
|--------|--------|------------|
| **Setup Time** | 5 minutes | 30-60 minutes |
| **Dependency Management** | ✅ Automatic | ❌ Manual |
| **Performance** | 99% native | 100% native |
| **Flexibility** | Limited | Full control |
| **Disk Space** | ~20GB | ~10GB |
| **Updates** | Pull new image | pip upgrade |
| **Best For** | Production, reproducibility | Development, customization |

**Recommendation**: Use **Docker** for production deployments. Use **Bare Metal** for active development or when you need to patch vLLM internals.

---

## Quick Start Summary

### Docker (5 minutes):
```bash
sudo docker run -d --gpus all --shm-size 8g -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf
```

### Bare Metal (30 minutes):
```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm
vllm serve meta-llama/Llama-2-7b-chat-hf
```

Both expose OpenAI-compatible API at `http://localhost:8000/v1/`.

---

## References

- vLLM Documentation: https://docs.vllm.ai
- vLLM GitHub: https://github.com/vllm-project/vllm
- Dockerfile source: `vllm/docker/Dockerfile`
- Server CLI: `vllm/entrypoints/cli/serve.py`
