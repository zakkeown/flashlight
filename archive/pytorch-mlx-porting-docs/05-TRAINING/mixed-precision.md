# Mixed Precision Training (AMP)

## Overview

**Automatic Mixed Precision (AMP)** is a technique that uses both float16 (FP16) and float32 (FP32) datatypes during training to:
- **Speed up training** by 2-4x (on modern GPUs with Tensor Cores)
- **Reduce memory usage** by ~50% (FP16 uses half the memory of FP32)
- **Maintain model accuracy** through selective precision and loss scaling

PyTorch's AMP implementation provides two key components:
1. **`torch.autocast`**: Automatic dtype selection for operations
2. **`torch.amp.GradScaler`**: Loss scaling to prevent gradient underflow

**File Locations**:
- Autocast: [torch/amp/autocast_mode.py](reference/pytorch/torch/amp/autocast_mode.py) (~400 lines)
- GradScaler: [torch/amp/grad_scaler.py](reference/pytorch/torch/amp/grad_scaler.py) (~600 lines)

---

## Key Concepts

### 1. Why Mixed Precision?

**Problem**: Float32 (FP32) is the default dtype in deep learning, but it's often overkill:
- **Memory**: FP32 uses 4 bytes per value
- **Compute**: FP32 operations are slower on modern hardware
- **Precision**: Most neural networks don't need full FP32 precision

**Solution**: Use float16 (FP16) for most operations:
- **Memory**: FP16 uses 2 bytes per value (50% reduction)
- **Compute**: Modern GPUs (V100, A100, H100) have specialized FP16/BF16 cores (Tensor Cores)
- **Speedup**: 2-4x faster training on supported hardware

### 2. Float16 Challenges

**Challenge 1: Limited Dynamic Range**
- **FP32**: Exponent range: 2^-126 to 2^127 (~10^-38 to 10^38)
- **FP16**: Exponent range: 2^-14 to 2^15 (~6×10^-5 to 6×10^4)

**Problem**: Gradients often fall outside FP16 range, leading to:
- **Underflow**: Small gradients become zero
- **Overflow**: Large values become inf/NaN

**Challenge 2: Precision Loss**
- **FP16 Mantissa**: 10 bits (vs 23 bits in FP32)
- **Problem**: Accumulated rounding errors in optimizer updates

### 3. AMP Solutions

**Solution 1: Selective Precision (Autocast)**
- Run **memory-bandwidth-bound ops** in FP16 (matmul, conv)
- Run **numerically-sensitive ops** in FP32 (softmax, loss, normalization)
- Automatically cast tensors to appropriate dtype

**Solution 2: Loss Scaling (GradScaler)**
- **Scale up loss** before backward pass to shift gradients into FP16 range
- **Unscale gradients** before optimizer step
- **Dynamically adjust scale** to avoid overflow/underflow

**Solution 3: Master Weights**
- Store optimizer state (momentum, variance) in FP32
- Perform parameter updates in FP32
- Convert back to FP16 for forward pass

---

## 1. Autocast Context Manager

### Description

`torch.autocast` automatically selects the appropriate precision for each operation. Operations run in an op-specific dtype chosen to balance performance and accuracy.

### API

```python
torch.autocast(
    device_type,            # 'cuda', 'cpu', 'mps', etc.
    dtype=None,             # torch.float16 or torch.bfloat16 (default varies by device)
    enabled=True,           # Enable/disable autocast
    cache_enabled=True      # Cache autocast dtype decisions
)
```

**Default dtypes**:
- **CUDA**: `torch.float16`
- **CPU**: `torch.bfloat16` (better numerical stability)
- **MPS** (Apple Silicon): `torch.float16`

### Implementation

From [autocast_mode.py:52-386](reference/pytorch/torch/amp/autocast_mode.py#L52-L386):

```python
class autocast:
    def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
        if not isinstance(device_type, str):
            raise ValueError(f"Expected `device_type` of type `str`, got: `{type(device_type)}`")

        self.fast_dtype = torch.get_autocast_dtype(device_type) if dtype is None else dtype
        self.device = device_type
        self._enabled = enabled
        self._cache_enabled = cache_enabled if cache_enabled is not None else torch.is_autocast_cache_enabled()

    def __enter__(self):
        # Save previous autocast state
        self.prev = torch.is_autocast_enabled(self.device)
        self.prev_fastdtype = torch.get_autocast_dtype(self.device)
        self.prev_cache_enabled = torch.is_autocast_cache_enabled()

        # Set new autocast state
        torch.set_autocast_enabled(self.device, self._enabled)
        torch.set_autocast_dtype(self.device, self.fast_dtype)
        torch.autocast_increment_nesting()
        torch.set_autocast_cache_enabled(self._cache_enabled)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous autocast state
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        torch.set_autocast_enabled(self.device, self.prev)
        torch.set_autocast_dtype(self.device, self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        return False
```

### Autocast Operation Rules

PyTorch categorizes operations into three types:

**1. FP16-Safe Operations** (run in FP16):
- `torch.mm`, `torch.matmul`, `torch.bmm` (matrix multiplication)
- `torch.conv1d`, `torch.conv2d`, `torch.conv3d` (convolutions)
- `torch.linear` (linear layers)
- `torch.addmm`, `torch.baddbmm`

**Why FP16**: These are memory-bandwidth-bound, benefit from reduced memory traffic.

**2. FP32-Required Operations** (run in FP32):
- `torch.softmax`, `torch.log_softmax` (numerically sensitive)
- `torch.nn.functional.cross_entropy`, `torch.nn.functional.nll_loss`
- `torch.sum`, `torch.mean` (accumulation ops)
- `torch.norm`, `torch.normalize`
- `torch.batch_norm`, `torch.layer_norm` (normalization)

**Why FP32**: These require high precision to avoid numerical instability.

**3. Promote-to-Widest Operations** (match input precision):
- `torch.add`, `torch.sub`, `torch.mul`, `torch.div`
- `torch.cat`, `torch.stack`
- Indexing, slicing operations

**Why Mixed**: Preserve input precision to avoid unnecessary conversions.

### Usage Example

**Basic Usage**:
```python
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for input, target in data_loader:
    optimizer.zero_grad()

    # Autocast only the forward pass
    with torch.autocast(device_type="cuda"):
        output = model(input)
        loss = loss_fn(output, target)

    # Backward pass runs in the same dtype as forward ops
    loss.backward()
    optimizer.step()
```

**As a Decorator**:
```python
class MyModel(nn.Module):
    @torch.autocast(device_type="cuda")
    def forward(self, x):
        return self.layers(x)
```

---

## 2. GradScaler

### Description

`GradScaler` prevents gradient underflow by scaling the loss before backward pass, then unscaling gradients before optimizer step. It also dynamically adjusts the scale factor to avoid overflow.

### Algorithm

```
1. Scale loss:
   scaled_loss = loss * scale_factor

2. Backward pass:
   scaled_loss.backward()  # Produces scaled_gradients

3. Unscale gradients:
   gradients = scaled_gradients / scale_factor

4. Check for inf/NaN:
   if any gradient contains inf/NaN:
       skip optimizer step
       scale_factor *= backoff_factor (e.g., 0.5)
   else:
       optimizer.step()
       if no inf/NaN for growth_interval steps:
           scale_factor *= growth_factor (e.g., 2.0)
```

### API

```python
torch.amp.GradScaler(
    device='cuda',              # Device type
    init_scale=2.**16,          # Initial scale (65536)
    growth_factor=2.0,          # Multiply scale by this when stable
    backoff_factor=0.5,         # Multiply scale by this on overflow
    growth_interval=2000,       # Steps before growing scale
    enabled=True                # Enable/disable scaling
)
```

### Implementation

From [grad_scaler.py:53-400](reference/pytorch/torch/amp/grad_scaler.py#L53-L400):

```python
class GradScaler:
    def __init__(self, device='cuda', init_scale=2.**16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._device = device
        self._enabled = enabled

        if enabled:
            self._init_scale = init_scale
            self._scale = None  # Lazily initialized
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            self._growth_tracker = None  # Lazily initialized
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def scale(self, outputs):
        """Multiply outputs by scale factor"""
        if not self._enabled:
            return outputs

        if isinstance(outputs, torch.Tensor):
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Handle iterable of tensors
        # ... (complex multi-device logic)

    def unscale_(self, optimizer):
        """Divide gradients by scale factor"""
        if not self._enabled:
            return

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state['stage'] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer")

        # Compute inverse scale in FP64 for precision
        inv_scale = self._scale.double().reciprocal().float()
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

        # Unscale gradients and check for inf/NaN
        optimizer_state['found_inf_per_device'] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, allow_fp16=False
        )
        optimizer_state['stage'] = OptState.UNSCALED

    def step(self, optimizer, *args, **kwargs):
        """Unscale gradients and call optimizer.step() if no inf/NaN"""
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        # Unscale if not already done
        if optimizer_state['stage'] is OptState.READY:
            self.unscale_(optimizer)

        # Check for inf/NaN across all devices
        found_inf = sum(v.item() for v in optimizer_state['found_inf_per_device'].values())

        # Only step if no inf/NaN
        retval = None
        if not found_inf:
            retval = optimizer.step(*args, **kwargs)

        optimizer_state['stage'] = OptState.STEPPED
        return retval

    def update(self, new_scale=None):
        """Update scale factor based on inf/NaN history"""
        if not self._enabled:
            return

        if new_scale is not None:
            self._scale.fill_(new_scale)
            return

        # Check if any optimizer found inf/NaN
        found_inf_combined = any(
            state.get('found_inf_per_device', {})
            for state in self._per_optimizer_states.values()
        )

        # Compute new scale
        if found_inf_combined:
            # Backoff: scale *= backoff_factor
            self._scale *= self._backoff_factor
            self._growth_tracker.fill_(0)
        else:
            # Increment growth tracker
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                # Grow: scale *= growth_factor
                self._scale *= self._growth_factor
                self._growth_tracker.fill_(0)

        # Reset optimizer states
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
```

### Key Implementation Details

**1. Lazy Initialization**:
```python
def _lazy_init_scale_growth_tracker(self, dev):
    self._scale = torch.full((), self._init_scale, dtype=torch.float32, device=dev)
    self._growth_tracker = torch.full((), 0, dtype=torch.int32, device=dev)
```
- Scale is initialized on first call to `scale()`
- Ensures scale tensor is on the correct device

**2. Unscaling with Inf/NaN Detection**:
```python
def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
    # Group gradients by device and dtype
    per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue

            grad = param.grad._values() if param.grad.is_sparse else param.grad
            per_device_and_dtype_grads[grad.device][grad.dtype].append(grad)

    # Unscale and check for inf/NaN per device/dtype
    for device, per_dtype_grads in per_device_and_dtype_grads.items():
        for grads in per_dtype_grads.values():
            torch._amp_foreach_non_finite_check_and_unscale_(
                grads,
                found_inf.to(device),
                inv_scale.to(device)
            )
```

**3. Optimizer State Tracking**:
```python
class OptState(Enum):
    READY = 0      # Ready for unscaling
    UNSCALED = 1   # Gradients have been unscaled
    STEPPED = 2    # Optimizer.step() has been called
```

### Usage Example

**Complete Training Loop**:
```python
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(device='cuda')

for epoch in range(epochs):
    for input, target in data_loader:
        optimizer.zero_grad()

        # Forward pass with autocast
        with torch.autocast(device_type='cuda'):
            output = model(input)
            loss = loss_fn(output, target)

        # Scale loss and backward
        scaler.scale(loss).backward()

        # Unscale gradients and step
        scaler.step(optimizer)

        # Update scale for next iteration
        scaler.update()
```

**With Gradient Clipping**:
```python
scaler = torch.amp.GradScaler(device='cuda')

for input, target in data_loader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda'):
        output = model(input)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()

    # Unscale gradients before clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

---

## 3. BFloat16 vs Float16

### Comparison

| Aspect | Float16 (FP16) | BFloat16 (BF16) |
|--------|----------------|-----------------|
| **Exponent bits** | 5 | 8 (same as FP32) |
| **Mantissa bits** | 10 | 7 |
| **Dynamic range** | 2^-14 to 2^15 | 2^-126 to 2^127 (same as FP32) |
| **Precision** | Higher | Lower |
| **Numerical stability** | Requires loss scaling | Often works without scaling |
| **Hardware support** | Volta, Turing, Ampere, Hopper | Ampere, Hopper, TPU, Apple Silicon |
| **Typical use** | CUDA GPUs | CPU, Apple Silicon, Google TPU |

### When to Use Each

**Float16**:
- ✅ NVIDIA GPUs (V100, A100, H100)
- ✅ Maximum memory savings
- ✅ Mature hardware support
- ⚠️ Requires GradScaler

**BFloat16**:
- ✅ CPU training (Intel, AMD)
- ✅ Apple Silicon (M1, M2, M3)
- ✅ Google TPUs
- ✅ Simpler training (often no scaling needed)
- ❌ Slightly lower precision

**CPU Example with BFloat16**:
```python
# CPU training with bfloat16
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for input, target in data_loader:
    optimizer.zero_grad()

    # BF16 autocast on CPU
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = model(input)
        loss = loss_fn(output, target)

    loss.backward()
    optimizer.step()
    # Note: No GradScaler needed for BF16!
```

---

## 4. Common Patterns

### Pattern 1: Multiple Losses

```python
scaler = torch.amp.GradScaler(device='cuda')

for input, target in data_loader:
    optimizer.zero_grad()

    with torch.autocast(device_type='cuda'):
        output = model(input)
        loss1 = loss_fn1(output, target)
        loss2 = loss_fn2(output, target)
        loss = loss1 + loss2

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Pattern 2: Gradient Accumulation

```python
scaler = torch.amp.GradScaler(device='cuda')
accumulation_steps = 4

for i, (input, target) in enumerate(data_loader):
    with torch.autocast(device_type='cuda'):
        output = model(input)
        loss = loss_fn(output, target) / accumulation_steps

    scaler.scale(loss).backward()

    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Pattern 3: Multiple Optimizers

```python
scaler = torch.amp.GradScaler(device='cuda')

for input, target in data_loader:
    optimizer1.zero_grad()
    optimizer2.zero_grad()

    with torch.autocast(device_type='cuda'):
        output = model(input)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()

    # Step each optimizer separately
    scaler.step(optimizer1)
    scaler.step(optimizer2)

    scaler.update()
```

### Pattern 4: Custom Autograd Functions

```python
class CustomFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd
    def forward(ctx, input):
        # Runs in autocast dtype
        output = input * 2
        ctx.save_for_backward(input)
        return output

    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        # Runs in same dtype as forward
        input, = ctx.saved_tensors
        return grad_output * 2
```

---

## 5. Performance Benchmarks

### Memory Usage

For ResNet-50 (25M parameters, batch size 256):

| Precision | Model Weights | Activations | Gradients | Total |
|-----------|---------------|-------------|-----------|-------|
| **FP32** | 100 MB | 8 GB | 100 MB | ~8.2 GB |
| **FP16** | 50 MB | 4 GB | 50 MB | ~4.1 GB |
| **Savings** | 50% | 50% | 50% | **50%** |

### Training Speed

On NVIDIA A100 GPU (ResNet-50, ImageNet):

| Configuration | Throughput | Speedup |
|---------------|------------|---------|
| FP32 (baseline) | 1,200 img/sec | 1.0x |
| FP16 (AMP) | 4,800 img/sec | **4.0x** |
| BF16 (AMP) | 4,500 img/sec | 3.75x |

**Note**: Speedup varies by:
- Model architecture (Transformers get 3-4x, CNNs get 2-3x)
- GPU generation (Ampere/Hopper get best speedup)
- Batch size (larger batches see more benefit)

### Accuracy

Most models train to similar accuracy with AMP:

| Model | FP32 Accuracy | FP16 (AMP) Accuracy | Δ |
|-------|---------------|---------------------|---|
| ResNet-50 | 76.15% | 76.12% | -0.03% |
| BERT-Large | 84.6 F1 | 84.5 F1 | -0.1 F1 |
| GPT-2 | 29.4 PPL | 29.5 PPL | +0.1 PPL |

---

## 6. Troubleshooting

### Problem 1: NaN/Inf Gradients

**Symptoms**: Loss becomes NaN, gradients explode

**Causes**:
1. Initial scale too high
2. Learning rate too high
3. Numerically unstable operation in FP16

**Solutions**:
```python
# Reduce initial scale
scaler = torch.amp.GradScaler(init_scale=2.**10)  # Default: 2.**16

# Disable autocast for unstable op
with torch.autocast(device_type='cuda', enabled=False):
    unstable_output = unstable_op(x.float())
```

### Problem 2: Loss Not Decreasing

**Symptoms**: Training doesn't converge, loss plateaus

**Cause**: Optimizer steps are being skipped due to inf/NaN

**Solution**:
```python
# Monitor scale and skipped steps
if (step + 1) % 100 == 0:
    print(f"Scale: {scaler.get_scale()}")
    print(f"Growth tracker: {scaler._growth_tracker}")
```

### Problem 3: No Speedup

**Symptoms**: FP16 training is not faster than FP32

**Causes**:
1. GPU doesn't have Tensor Cores (pre-Volta)
2. Model is too small (overhead dominates)
3. Operations not running in FP16

**Solutions**:
```python
# Check if ops are running in FP16
with torch.autocast(device_type='cuda'):
    x = torch.randn(1024, 1024, device='cuda')
    y = torch.mm(x, x)
    print(y.dtype)  # Should print torch.float16

# Use larger batch sizes to amortize overhead
```

---

## MLX Porting Guide

### Recommended Priority

**High Priority**:
1. **Autocast** - Essential for mixed-precision training
2. **Basic GradScaler** - Loss scaling without dynamic adjustment

**Medium Priority**:
3. **Dynamic Scaling** - Automatic scale adjustment
4. **BFloat16 Support** - Apple Silicon optimization

### MLX C++ API

```cpp
namespace mlx::nn {

// Autocast context manager
class AutocastMode {
 public:
  AutocastMode(Dtype dtype, bool enabled = true)
      : dtype_(dtype), enabled_(enabled) {}

  void enter() {
    if (!enabled_) return;
    prev_dtype_ = get_autocast_dtype();
    prev_enabled_ = is_autocast_enabled();
    set_autocast_dtype(dtype_);
    set_autocast_enabled(true);
  }

  void exit() {
    if (!enabled_) return;
    set_autocast_dtype(prev_dtype_);
    set_autocast_enabled(prev_enabled_);
  }

 private:
  Dtype dtype_;
  bool enabled_;
  Dtype prev_dtype_;
  bool prev_enabled_;
};

// Gradient scaler
class GradScaler {
 public:
  GradScaler(float init_scale = 65536.0f,
             float growth_factor = 2.0f,
             float backoff_factor = 0.5f,
             int growth_interval = 2000)
      : scale_(init_scale),
        growth_factor_(growth_factor),
        backoff_factor_(backoff_factor),
        growth_interval_(growth_interval),
        growth_tracker_(0) {}

  array scale(const array& loss) {
    return loss * scale_;
  }

  void unscale(std::unordered_map<std::string, array>& gradients) {
    float inv_scale = 1.0f / scale_;
    for (auto& [name, grad] : gradients) {
      grad = grad * inv_scale;
      // Check for inf/NaN
      if (any(isinf(grad) || isnan(grad)).item<bool>()) {
        found_inf_ = true;
      }
    }
  }

  bool step(Optimizer& optimizer,
            std::unordered_map<std::string, array>& parameters,
            std::unordered_map<std::string, array>& gradients) {
    unscale(gradients);

    if (found_inf_) {
      scale_ *= backoff_factor_;
      growth_tracker_ = 0;
      found_inf_ = false;
      return false;  // Skip optimizer step
    }

    optimizer.update(parameters, gradients);
    return true;
  }

  void update() {
    if (!found_inf_) {
      growth_tracker_++;
      if (growth_tracker_ >= growth_interval_) {
        scale_ *= growth_factor_;
        growth_tracker_ = 0;
      }
    }
  }

 private:
  float scale_;
  float growth_factor_;
  float backoff_factor_;
  int growth_interval_;
  int growth_tracker_;
  bool found_inf_ = false;
};

}  // namespace mlx::nn
```

### MLX Python API

```python
import mlx.core as mx

class autocast:
    """MLX autocast context manager"""

    def __init__(self, dtype=mx.float16, enabled=True):
        self.dtype = dtype
        self.enabled = enabled
        self.prev_dtype = None
        self.prev_enabled = None

    def __enter__(self):
        if not self.enabled:
            return self
        self.prev_dtype = mx.get_autocast_dtype()
        self.prev_enabled = mx.is_autocast_enabled()
        mx.set_autocast_dtype(self.dtype)
        mx.set_autocast_enabled(True)
        return self

    def __exit__(self, *args):
        if not self.enabled:
            return
        mx.set_autocast_dtype(self.prev_dtype)
        mx.set_autocast_enabled(self.prev_enabled)

class GradScaler:
    """MLX gradient scaler"""

    def __init__(self, init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0
        self.enabled = enabled
        self.found_inf = False

    def scale_loss(self, loss):
        if not self.enabled:
            return loss
        return loss * self.scale

    def unscale_gradients(self, gradients):
        if not self.enabled:
            return

        inv_scale = 1.0 / self.scale
        self.found_inf = False

        for key in gradients:
            gradients[key] = gradients[key] * inv_scale
            if mx.any(mx.isinf(gradients[key]) | mx.isnan(gradients[key])):
                self.found_inf = True

    def step(self, optimizer, parameters, gradients):
        self.unscale_gradients(gradients)

        if self.found_inf:
            self.scale *= self.backoff_factor
            self.growth_tracker = 0
            return False

        optimizer.update(parameters, gradients)
        return True

    def update(self):
        if not self.enabled:
            return

        if not self.found_inf:
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self.growth_tracker = 0
```

### Usage Example (MLX)

```python
import mlx.core as mx
import mlx.nn as nn

model = MyModel()
optimizer = nn.optimizers.AdamW(learning_rate=1e-3)
scaler = GradScaler()

def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for X_batch, y_batch in data_loader:
    # Forward pass with autocast
    with autocast(dtype=mx.float16):
        loss, grads = loss_and_grad_fn(model, X_batch, y_batch)

    # Scale loss and gradients
    scaled_loss = scaler.scale_loss(loss)

    # Step optimizer (handles unscaling and inf/NaN check)
    scaler.step(optimizer, model.parameters(), grads)

    # Update scale
    scaler.update()
```

---

## References

1. **Mixed Precision Training**: Micikevicius, P. et al. (2017). "Mixed Precision Training"
2. **Automatic Mixed Precision**: NVIDIA (2019). "Training With Mixed Precision"
3. **BFloat16**: Google (2019). "A Study of BFLOAT16 for Deep Learning Training"

---

## Summary

**Automatic Mixed Precision (AMP) Benefits**:
- ✅ 2-4x faster training on modern GPUs
- ✅ 50% memory reduction
- ✅ Minimal accuracy loss (<0.1% typically)
- ✅ Simple API (2 lines of code)

**Key Components**:
1. **`torch.autocast`**: Automatic dtype selection per operation
2. **`torch.amp.GradScaler`**: Loss scaling to prevent gradient underflow

**Best Practices**:
- Use FP16 on NVIDIA GPUs (with GradScaler)
- Use BF16 on CPU and Apple Silicon (often without scaling)
- Autocast only the forward pass, not backward
- Unscale gradients before clipping
- Monitor scale factor to detect training issues

**MLX Implementation Priority**: **High** - Essential for efficient training on Apple Silicon (M1/M2/M3). Implement autocast and basic GradScaler first, then add dynamic scaling.
