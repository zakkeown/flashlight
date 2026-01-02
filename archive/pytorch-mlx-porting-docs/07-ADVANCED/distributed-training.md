# Distributed Training

## Overview

PyTorch's distributed training infrastructure enables scaling training across multiple GPUs, machines, and nodes. The `torch.distributed` package provides collective communication primitives and the `DistributedDataParallel` (DDP) module for synchronized gradient updates.

**Reference Files:**
- `torch/distributed/distributed_c10d.py` - Collective operations
- `torch/nn/parallel/distributed.py` - DistributedDataParallel

## Architecture

```
Distributed Training Components
├── Process Groups           - Logical groupings of processes
├── Backends                 - Communication implementations (NCCL, Gloo, MPI)
├── Collective Operations    - Synchronized multi-process ops
├── Point-to-Point           - Direct process communication
└── DistributedDataParallel  - High-level training wrapper
```

---

## Process Group Initialization

### init_process_group()

Initializes the default distributed process group.

```python
def init_process_group(
    backend: str | None = None,        # Backend: 'nccl', 'gloo', 'mpi', 'ucc'
    init_method: str | None = None,    # URL for peer discovery
    timeout: timedelta | None = None,  # Operation timeout
    world_size: int = -1,              # Total number of processes
    rank: int = -1,                    # This process's rank (0 to world_size-1)
    store: Store | None = None,        # Key/value store for coordination
    group_name: str = "",              # Deprecated
    pg_options: Any | None = None,     # Backend-specific options
    device_id: torch.device | int | None = None,  # Device for this process
) -> None
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `backend` | Communication backend: 'nccl' (GPU), 'gloo' (CPU/GPU), 'mpi' |
| `init_method` | URL for peer discovery: 'env://', 'tcp://host:port', 'file://path' |
| `world_size` | Total number of participating processes |
| `rank` | Unique ID for this process (0 to world_size-1) |
| `store` | Distributed key-value store for coordination |
| `timeout` | Default: 10 min (NCCL), 30 min (others) |
| `device_id` | GPU device for this process (enables optimizations) |

### Backends

| Backend | Use Case | Features |
|---------|----------|----------|
| `nccl` | GPU training | Fastest for CUDA, recommended for multi-GPU |
| `gloo` | CPU training, fallback | Cross-platform, supports CPU tensors |
| `mpi` | HPC clusters | Requires MPI installation |
| `ucc` | Experimental | Unified collective communications |

### Initialization Methods

```python
import torch.distributed as dist

# Method 1: Environment variables (most common with launchers)
# Expects: MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK
dist.init_process_group(backend='nccl', init_method='env://')

# Method 2: TCP with explicit address
dist.init_process_group(
    backend='nccl',
    init_method='tcp://192.168.1.1:23456',
    world_size=4,
    rank=0
)

# Method 3: Shared file system
dist.init_process_group(
    backend='gloo',
    init_method='file:///mnt/nfs/sharedfile',
    world_size=4,
    rank=0
)

# Method 4: Store-based (programmatic control)
store = dist.TCPStore("192.168.1.1", 23456, world_size, is_master=(rank == 0))
dist.init_process_group(backend='nccl', store=store, world_size=4, rank=0)
```

---

## Reduce Operations

```python
class ReduceOp:
    SUM      # Element-wise sum
    AVG      # Element-wise average
    PRODUCT  # Element-wise product
    MIN      # Element-wise minimum
    MAX      # Element-wise maximum
    BAND     # Bitwise AND
    BOR      # Bitwise OR
    BXOR     # Bitwise XOR
    PREMUL_SUM  # Pre-multiplied sum
```

---

## Collective Operations

### all_reduce()

Reduces tensor data across all processes so all get the final result.

```python
def all_reduce(
    tensor: Tensor,              # Input/output tensor (in-place)
    op: ReduceOp = ReduceOp.SUM, # Reduction operation
    group: ProcessGroup = None,  # Process group
    async_op: bool = False       # Async operation
) -> Work | None
```

**Example:**

```python
import torch.distributed as dist

# Each process has different data
# Rank 0: tensor([1, 2])
# Rank 1: tensor([3, 4])
tensor = torch.arange(2) + 1 + 2 * rank

dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# After: all processes have tensor([4, 6])
```

**Visual:**

```
Before:           After all_reduce(SUM):
Rank 0: [1, 2]    Rank 0: [4, 6]
Rank 1: [3, 4]    Rank 1: [4, 6]
                  (1+3=4, 2+4=6)
```

### broadcast()

Broadcasts tensor from source rank to all other processes.

```python
def broadcast(
    tensor: Tensor,              # Data to send (src) or receive buffer (others)
    src: int,                    # Source rank (global)
    group: ProcessGroup = None,
    async_op: bool = False
) -> Work | None
```

**Example:**

```python
if rank == 0:
    tensor = torch.tensor([1, 2, 3, 4])
else:
    tensor = torch.zeros(4)

dist.broadcast(tensor, src=0)
# All ranks now have tensor([1, 2, 3, 4])
```

**Visual:**

```
Before:           After broadcast(src=0):
Rank 0: [1,2,3,4] Rank 0: [1,2,3,4]
Rank 1: [0,0,0,0] Rank 1: [1,2,3,4]
Rank 2: [0,0,0,0] Rank 2: [1,2,3,4]
```

### all_gather()

Gathers tensors from all processes into a list on every process.

```python
def all_gather(
    tensor_list: list[Tensor],   # Output list (pre-allocated)
    tensor: Tensor,              # This process's input tensor
    group: ProcessGroup = None,
    async_op: bool = False
) -> Work | None
```

**Example:**

```python
# Each rank prepares output list
tensor_list = [torch.zeros(2) for _ in range(world_size)]
tensor = torch.arange(2) + 1 + 2 * rank

dist.all_gather(tensor_list, tensor)

# All ranks have: [tensor([1, 2]), tensor([3, 4])]
```

**Visual:**

```
Before:           After all_gather:
Rank 0: [1, 2]    Rank 0: [[1,2], [3,4]]
Rank 1: [3, 4]    Rank 1: [[1,2], [3,4]]
```

### reduce_scatter()

Reduces input list and scatters to all processes.

```python
def reduce_scatter(
    output: Tensor,              # Output tensor
    input_list: list[Tensor],    # List of tensors to reduce
    op: ReduceOp = ReduceOp.SUM,
    group: ProcessGroup = None,
    async_op: bool = False
) -> Work | None
```

**Example:**

```python
# 2 ranks, each provides 2 chunks
input_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]  # Same on all ranks
output = torch.zeros(2)

dist.reduce_scatter(output, input_list)

# Rank 0 gets sum of first chunks: [2, 4]  (1+1, 2+2)
# Rank 1 gets sum of second chunks: [6, 8] (3+3, 4+4)
```

**Visual:**

```
Input (same on both):   After reduce_scatter(SUM):
Rank 0: [[1,2], [3,4]]  Rank 0: [2, 4]   (sum of index 0)
Rank 1: [[1,2], [3,4]]  Rank 1: [6, 8]   (sum of index 1)
```

### barrier()

Synchronizes all processes - blocks until all reach this point.

```python
def barrier(
    group: ProcessGroup = None,
    async_op: bool = False,
    device_ids: list[int] = None
) -> Work | None
```

**Example:**

```python
# Wait for all processes before continuing
dist.barrier()
print(f"Rank {rank} passed barrier")
```

---

## Point-to-Point Operations

### send() / recv()

Blocking point-to-point communication.

```python
def send(tensor: Tensor, dst: int, group=None, tag: int = 0) -> None
def recv(tensor: Tensor, src: int = None, group=None, tag: int = 0) -> int
```

### isend() / irecv()

Non-blocking point-to-point communication.

```python
def isend(tensor: Tensor, dst: int, group=None, tag: int = 0) -> Work
def irecv(tensor: Tensor, src: int = None, group=None, tag: int = 0) -> Work
```

**Example:**

```python
if rank == 0:
    tensor = torch.tensor([1, 2, 3])
    dist.send(tensor, dst=1)
elif rank == 1:
    tensor = torch.zeros(3)
    dist.recv(tensor, src=0)
    # tensor is now [1, 2, 3]
```

---

## DistributedDataParallel (DDP)

High-level wrapper for distributed data parallel training.

### Class Definition

```python
class DistributedDataParallel(Module):
    def __init__(
        self,
        module: Module,                    # Model to wrap
        device_ids: list[int] = None,      # GPU devices for this replica
        output_device: int = None,         # Device for outputs
        dim: int = 0,                      # Batch dimension
        broadcast_buffers: bool = True,    # Sync buffers each forward
        process_group: ProcessGroup = None,
        bucket_cap_mb: float = 25,         # Gradient bucket size (MB)
        find_unused_parameters: bool = False,  # Handle unused params
        gradient_as_bucket_view: bool = False, # Memory optimization
        static_graph: bool = False,        # Graph doesn't change
    )
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `module` | The model to parallelize |
| `device_ids` | List of GPU IDs (single element for typical use) |
| `broadcast_buffers` | Sync BatchNorm stats etc. from rank 0 |
| `find_unused_parameters` | Required if some params don't get gradients |
| `gradient_as_bucket_view` | Reduces memory by avoiding gradient copies |
| `static_graph` | Optimization when computation graph is fixed |

### Basic Usage

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and move to GPU
    model = MyModel().to(rank)

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # Use DistributedSampler for data loading
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    optimizer = torch.optim.Adam(ddp_model.parameters())

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch

        for batch in dataloader:
            optimizer.zero_grad()
            loss = ddp_model(batch)
            loss.backward()  # Gradients synchronized automatically
            optimizer.step()

    cleanup()

# Launch with torch.multiprocessing.spawn or torchrun
```

### How DDP Works

1. **Initialization**: Model replicated on each GPU, parameters synchronized
2. **Forward**: Each GPU processes its batch independently
3. **Backward**: Gradients computed locally
4. **All-Reduce**: Gradients averaged across all GPUs (overlapped with backward)
5. **Update**: Optimizer step with synchronized gradients

```
GPU 0                    GPU 1
  │                        │
  ▼                        ▼
Forward(batch_0)        Forward(batch_1)
  │                        │
  ▼                        ▼
Backward                 Backward
  │                        │
  └──────all_reduce───────┘
           │
           ▼
    Averaged Gradients
           │
    ┌──────┴──────┐
    ▼             ▼
 Update        Update
```

### Saving/Loading DDP Models

```python
# Save (from any rank, usually rank 0)
if rank == 0:
    # Save the underlying module, not the DDP wrapper
    torch.save(ddp_model.module.state_dict(), 'model.pt')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model = model.to(rank)
ddp_model = DDP(model, device_ids=[rank])
```

---

## Process Groups

### Creating Sub-Groups

```python
# Split world into groups
world_size = dist.get_world_size()
rank = dist.get_rank()

# Create group with specific ranks
new_group = dist.new_group([0, 1, 2])  # Only ranks 0, 1, 2

# Operations on subgroup
if rank in [0, 1, 2]:
    dist.all_reduce(tensor, group=new_group)
```

### Utility Functions

```python
dist.get_rank()        # Current process rank
dist.get_world_size()  # Total number of processes
dist.is_initialized()  # Check if initialized
dist.get_backend()     # Current backend name
dist.destroy_process_group()  # Cleanup
```

---

## Launching Distributed Training

### Using torchrun (Recommended)

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    train.py
```

### Using torch.multiprocessing.spawn

```python
import torch.multiprocessing as mp

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
```

---

## Async Operations

```python
# Async all-reduce
work = dist.all_reduce(tensor, async_op=True)

# Do other work while communication happens
# ...

# Wait for completion
work.wait()

# Check if done without blocking
if work.is_completed():
    print("Done!")
```

---

## Common Patterns

### Gradient Accumulation with DDP

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    # Disable gradient sync except on accumulation boundaries
    with ddp_model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext():
        loss = ddp_model(batch) / accumulation_steps
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision with DDP

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast(device_type='cuda'):
        loss = ddp_model(batch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## MLX Mapping

### Multi-Device in MLX

MLX's unified memory architecture on Apple Silicon differs fundamentally from distributed CUDA training. MLX focuses on single-node efficiency rather than multi-node distribution.

```python
import mlx.core as mx

# MLX uses unified memory - no explicit device management
x = mx.array([1, 2, 3])

# Data parallelism patterns differ
# MLX emphasizes batch parallelism via vectorized operations
```

### Conceptual Mapping

| PyTorch Distributed | MLX Equivalent |
|---------------------|----------------|
| `init_process_group` | Not applicable (single process) |
| `all_reduce` | Direct array operations |
| `DistributedDataParallel` | Standard model (unified memory) |
| `DistributedSampler` | Standard sampling |
| Multi-GPU training | Single unified memory pool |

### Training Patterns

```python
# MLX training - simpler without distribution
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def train_step(model, x, y, optimizer):
    def loss_fn(model):
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss, grads = mx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

# Compile for efficiency
train_step = mx.compile(train_step)
```

### Key Differences

| Aspect | PyTorch Distributed | MLX |
|--------|---------------------|-----|
| Memory Model | Distributed across GPUs | Unified memory |
| Communication | Explicit collectives | Not needed |
| Scaling | Multi-node clusters | Single machine optimization |
| Process Model | Multi-process | Single process |
| Gradient Sync | Explicit all-reduce | Automatic |

---

## Best Practices

1. **Use NCCL for GPU training** - Fastest backend for CUDA devices

2. **Set device before DDP** - `torch.cuda.set_device(rank)` before model creation

3. **Use DistributedSampler** - Ensures non-overlapping data across ranks

4. **Call sampler.set_epoch()** - Different shuffling each epoch

5. **Save from rank 0 only** - Avoid file conflicts

6. **Access underlying module** - `ddp_model.module` for model operations

7. **Handle unused parameters** - Set `find_unused_parameters=True` if needed

8. **Use gradient bucketing** - DDP automatically buckets for efficiency

9. **Consider static_graph** - Enables optimizations for fixed graphs

10. **Clean up resources** - Call `destroy_process_group()` when done

---

## Debugging

### Common Issues

**Hanging operations:**
- Check all ranks reach the same collectives
- Verify tensor shapes match across ranks
- Ensure same code path in all processes

**NCCL errors:**
- Set `NCCL_DEBUG=INFO` for verbose output
- Check network connectivity between nodes
- Verify CUDA devices are accessible

**Gradient mismatch:**
- Ensure same random seeds for model init
- Verify batch sizes are consistent
- Check for non-deterministic operations

```python
# Debug environment variables
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| `init_process_group()` | Initialize distributed runtime |
| `all_reduce()` | Reduce + broadcast to all |
| `broadcast()` | Send from one to all |
| `all_gather()` | Gather from all to all |
| `reduce_scatter()` | Reduce then scatter |
| `barrier()` | Synchronization point |
| `send()`/`recv()` | Point-to-point blocking |
| `isend()`/`irecv()` | Point-to-point non-blocking |
| `DistributedDataParallel` | High-level training wrapper |
| `DistributedSampler` | Non-overlapping data sampling |
