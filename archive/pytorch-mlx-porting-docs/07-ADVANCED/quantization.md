# Quantization

## Overview

Quantization reduces model size and improves inference speed by converting floating-point weights and activations to lower-precision integers (e.g., INT8, INT4). PyTorch provides comprehensive quantization support through post-training quantization (PTQ) and quantization-aware training (QAT).

**Reference Files:**
- `torch/ao/quantization/quantize.py` - Core quantization functions
- `torch/ao/quantization/observer.py` - Statistics collection
- `torch/ao/quantization/fake_quantize.py` - Simulated quantization for training
- `torch/ao/quantization/qconfig.py` - Quantization configuration

## Quantization Types

```
Quantization Approaches
├── Post-Training Quantization (PTQ)
│   ├── Dynamic Quantization     - Weights static, activations dynamic
│   ├── Static Quantization      - Both weights and activations static
│   └── Weight-Only Quantization - Only weights quantized
├── Quantization-Aware Training (QAT)
│   └── Fake quantization during training
└── FX Graph Mode Quantization
    └── Graph-based automated quantization
```

---

## Quantization Math

### Affine Quantization (Asymmetric)

```
Q(x) = round(x / scale + zero_point)
x_dequant = (Q(x) - zero_point) * scale
```

Where:
- `scale = (x_max - x_min) / (quant_max - quant_min)`
- `zero_point = quant_min - round(x_min / scale)`

### Symmetric Quantization

```
Q(x) = round(x / scale)
x_dequant = Q(x) * scale
```

Where:
- `scale = max(|x_max|, |x_min|) / quant_max`
- `zero_point = 0`

---

## Quantization Schemes

```python
# Per-tensor: single scale/zero_point for entire tensor
torch.per_tensor_affine
torch.per_tensor_symmetric

# Per-channel: different scale/zero_point per output channel
torch.per_channel_affine
torch.per_channel_symmetric
```

---

## Observers

Observers collect tensor statistics during calibration to compute quantization parameters.

### ObserverBase

```python
class ObserverBase(nn.Module):
    def __init__(self, dtype, is_dynamic=False):
        self.dtype = dtype          # Target quantized dtype
        self.is_dynamic = is_dynamic

    def forward(self, x):
        """Observe tensor and update statistics."""
        pass

    def calculate_qparams(self):
        """Return (scale, zero_point) based on collected stats."""
        pass
```

### Built-in Observers

| Observer | Description |
|----------|-------------|
| `MinMaxObserver` | Tracks global min/max |
| `MovingAverageMinMaxObserver` | Exponential moving average of min/max |
| `PerChannelMinMaxObserver` | Per-channel min/max tracking |
| `HistogramObserver` | Histogram-based for optimal range |
| `PlaceholderObserver` | Dynamic quantization placeholder |

### MinMaxObserver

```python
class MinMaxObserver(UniformQuantizationObserverBase):
    def __init__(
        self,
        dtype=torch.quint8,              # Target dtype
        qscheme=torch.per_tensor_affine, # Quantization scheme
        reduce_range=False,              # Reduce by 1 bit
        quant_min=None,                  # Override min value
        quant_max=None,                  # Override max value
    ):
        ...

    def forward(self, x):
        # Update min_val and max_val
        min_val = torch.min(x)
        max_val = torch.max(x)
        self.min_val = torch.min(self.min_val, min_val)
        self.max_val = torch.max(self.max_val, max_val)
        return x

    def calculate_qparams(self):
        # Compute scale and zero_point from min_val, max_val
        scale = (self.max_val - self.min_val) / (quant_max - quant_min)
        zero_point = quant_min - torch.round(self.min_val / scale)
        return scale, zero_point
```

### Usage

```python
from torch.ao.quantization.observer import MinMaxObserver

observer = MinMaxObserver(dtype=torch.qint8)

# Calibrate with representative data
for batch in calibration_data:
    observer(batch)

# Get quantization parameters
scale, zero_point = observer.calculate_qparams()
```

---

## FakeQuantize

Simulates quantization during forward pass while maintaining gradients for backpropagation (used in QAT).

### FakeQuantize Class

```python
class FakeQuantize(nn.Module):
    """
    Output:
        x_out = (clamp(round(x/scale + zp), qmin, qmax) - zp) * scale
    """

    def __init__(
        self,
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    ):
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.activation_post_process = observer()
        self.fake_quant_enabled = True
        self.observer_enabled = True

    def forward(self, x):
        if self.observer_enabled:
            self.activation_post_process(x)

        if self.fake_quant_enabled:
            scale, zero_point = self.calculate_qparams()
            x = torch.fake_quantize_per_tensor_affine(
                x, scale, zero_point, self.quant_min, self.quant_max
            )
        return x

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled = enabled

    def enable_observer(self, enabled=True):
        self.observer_enabled = enabled
```

### Built-in FakeQuantize Modules

```python
from torch.ao.quantization.fake_quantize import (
    default_fake_quant,              # Activations
    default_weight_fake_quant,       # Weights
    default_per_channel_weight_fake_quant,  # Per-channel weights
    default_dynamic_fake_quant,      # Dynamic quantization
)
```

---

## QConfig

Combines observer/fake_quantize for activations and weights.

### QConfig Structure

```python
class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Configuration for quantizing a layer.

    Args:
        activation: Observer class for activations
        weight: Observer class for weights
    """
    pass
```

### Built-in QConfigs

```python
from torch.ao.quantization import (
    default_qconfig,           # Standard 8-bit PTQ
    default_per_channel_qconfig,  # Per-channel weights
    default_dynamic_qconfig,   # Dynamic quantization
    default_qat_qconfig,       # QAT config
    float16_dynamic_qconfig,   # FP16 weights, dynamic activations
)
```

### Custom QConfig

```python
from torch.ao.quantization import QConfig
from torch.ao.quantization.observer import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
)

my_qconfig = QConfig(
    activation=MinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    ),
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric
    )
)
```

---

## Post-Training Quantization (PTQ)

### Dynamic Quantization

Weights quantized statically, activations quantized dynamically at runtime.

```python
import torch.ao.quantization as quant

# Original model
model = MyModel()
model.eval()

# Apply dynamic quantization
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear, torch.nn.LSTM},  # Layers to quantize
    dtype=torch.qint8  # Target dtype
)
```

### Static Quantization

Both weights and activations quantized using calibration data.

```python
import torch.ao.quantization as quant

# Step 1: Prepare model
model = MyModel()
model.eval()

# Insert observers
model.qconfig = quant.default_qconfig
quant.prepare(model, inplace=True)

# Step 2: Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:
        model(batch)

# Step 3: Convert to quantized model
quant.convert(model, inplace=True)
```

### Workflow Diagram

```
Static PTQ Workflow:
┌─────────────┐    ┌────────────┐    ┌─────────────┐    ┌────────────┐
│ Float Model │ -> │  prepare() │ -> │  Calibrate  │ -> │  convert() │
└─────────────┘    └────────────┘    └─────────────┘    └────────────┘
                   (add observers)   (run inference     (observers ->
                                      on data)           quantized ops)
```

---

## Quantization-Aware Training (QAT)

Simulates quantization during training to improve quantized model accuracy.

```python
import torch.ao.quantization as quant

# Step 1: Define model with QAT config
model = MyModel()
model.train()

model.qconfig = quant.get_default_qat_qconfig('fbgemm')  # or 'qnnpack'

# Step 2: Prepare for QAT (inserts FakeQuantize modules)
quant.prepare_qat(model, inplace=True)

# Step 3: Train with fake quantization
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    for batch, target in train_loader:
        optimizer.zero_grad()
        output = model(batch)  # FakeQuantize active
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Step 4: Convert to quantized model
model.eval()
quant.convert(model, inplace=True)
```

### QAT Workflow

```
QAT Workflow:
┌─────────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐
│ Float Model │ -> │ prepare_qat() │ -> │   Train   │ -> │  convert() │
└─────────────┘    └───────────────┘    └───────────┘    └────────────┘
                   (add FakeQuantize)   (learn to be     (to actual
                                         robust to        quantized)
                                         quantization)
```

---

## Core Functions

### prepare()

Inserts observers into the model.

```python
def prepare(
    model: nn.Module,
    inplace: bool = False,
    allow_list: set = None,
    observer_non_leaf_module_list: list = None,
    prepare_custom_config_dict: dict = None
) -> nn.Module
```

### convert()

Converts prepared model to quantized model.

```python
def convert(
    module: nn.Module,
    inplace: bool = False,
    mapping: dict = None,
    convert_custom_config_dict: dict = None
) -> nn.Module
```

### quantize()

Combines prepare + calibration + convert.

```python
def quantize(
    model: nn.Module,
    run_fn: Callable,      # Calibration function
    run_args: tuple,       # Args to run_fn
    inplace: bool = False
) -> nn.Module
```

---

## FX Graph Mode Quantization

More flexible quantization using FX graph capture.

```python
from torch.ao.quantization import quantize_fx

# Prepare
model = MyModel()
model.eval()

qconfig_mapping = get_default_qconfig_mapping("fbgemm")
example_inputs = (torch.randn(1, 3, 224, 224),)

# Prepare for calibration
prepared_model = quantize_fx.prepare_fx(
    model,
    qconfig_mapping,
    example_inputs
)

# Calibrate
with torch.no_grad():
    for batch in calibration_loader:
        prepared_model(batch)

# Convert
quantized_model = quantize_fx.convert_fx(prepared_model)
```

---

## Quantized Data Types

| dtype | Bits | Range | Use Case |
|-------|------|-------|----------|
| `torch.qint8` | 8 | [-128, 127] | Weights (signed) |
| `torch.quint8` | 8 | [0, 255] | Activations (unsigned) |
| `torch.qint32` | 32 | [-2³¹, 2³¹-1] | Biases, accumulators |
| `torch.quint4x2` | 4 | [0, 15] | Low-bit weights |
| `torch.float16` | 16 | FP range | Mixed precision |

---

## Quantized Modules

PyTorch provides quantized versions of common modules:

```python
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd

# Static quantized modules
nnq.Linear
nnq.Conv1d, nnq.Conv2d, nnq.Conv3d
nnq.BatchNorm2d
nnq.ReLU
nnq.Embedding

# Dynamic quantized modules
nnqd.Linear
nnqd.LSTM
nnqd.GRU
```

---

## Backends

| Backend | Platform | Features |
|---------|----------|----------|
| `fbgemm` | x86 CPUs | Optimized for server |
| `qnnpack` | ARM CPUs | Optimized for mobile |
| `onednn` | Intel CPUs | Intel-specific optimizations |

```python
# Set backend
torch.backends.quantized.engine = 'fbgemm'

# Or use backend-specific qconfig
qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
```

---

## Common Patterns

### Module Fusion

Fuse operations for better quantization.

```python
from torch.ao.quantization import fuse_modules

# Fuse Conv-BN-ReLU
model = fuse_modules(model, [['conv', 'bn', 'relu']])

# Fuse Linear-ReLU
model = fuse_modules(model, [['fc', 'relu']])
```

### QuantStub / DeQuantStub

Mark quantization boundaries.

```python
class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        x = self.quant(x)      # float -> quantized
        x = self.linear(x)
        x = self.dequant(x)    # quantized -> float
        return x
```

---

## MLX Mapping

### MLX Quantization

MLX supports quantization through its own API:

```python
import mlx.core as mx
import mlx.nn as nn

# MLX quantization functions
def quantize(w, group_size=64, bits=4):
    """Quantize weights to low-bit representation."""
    # Group quantization
    w_reshaped = w.reshape(-1, group_size)
    scales = mx.max(mx.abs(w_reshaped), axis=1, keepdims=True) / (2**(bits-1) - 1)
    w_quantized = mx.round(w_reshaped / scales)
    return w_quantized, scales

def dequantize(w_quantized, scales, group_size=64):
    """Dequantize back to float."""
    return (w_quantized * scales).reshape(original_shape)
```

### MLX QuantizedLinear

```python
import mlx.nn as nn

# MLX provides quantized layers
class QuantizedLinear(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ):
        ...

# Usage
layer = nn.QuantizedLinear(512, 512, bits=4, group_size=64)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| Quantization Types | PTQ, QAT, Dynamic | Primarily weight quantization |
| Bit Widths | 8-bit, 4-bit, 2-bit | 4-bit, 8-bit common |
| Observers | MinMax, Histogram, etc. | Simpler calibration |
| Workflow | prepare → calibrate → convert | Direct quantize/dequantize |
| Backend | fbgemm, qnnpack | Metal-optimized |

### Converting PyTorch Quantized to MLX

```python
def convert_pytorch_quantized_to_mlx(pytorch_model):
    """Convert PyTorch quantized model to MLX."""
    mlx_weights = {}

    for name, param in pytorch_model.state_dict().items():
        if 'weight' in name:
            # Dequantize PyTorch weights
            if hasattr(param, 'dequantize'):
                float_weight = param.dequantize()
            else:
                float_weight = param

            # Requantize for MLX
            mlx_weights[name] = mx.array(float_weight.numpy())

    return mlx_weights
```

---

## Best Practices

1. **Calibrate with representative data** - Use data similar to inference distribution

2. **Use per-channel for weights** - Better accuracy than per-tensor

3. **Fuse operations first** - Conv-BN-ReLU fusion improves accuracy

4. **Choose appropriate backend** - fbgemm for x86, qnnpack for ARM

5. **Validate accuracy** - Compare quantized vs float model outputs

6. **Use QAT for sensitive models** - When PTQ accuracy is insufficient

7. **Benchmark carefully** - Quantization benefits vary by model/hardware

8. **Watch for outliers** - Histogram observers handle outliers better

---

## Debugging

### Compare Quantized vs Float

```python
# Run both models
float_output = float_model(input)
quant_output = quantized_model(input)

# Compare
diff = (float_output - quant_output).abs()
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
```

### Inspect Quantization Parameters

```python
for name, module in quantized_model.named_modules():
    if hasattr(module, 'scale'):
        print(f"{name}: scale={module.scale}, zp={module.zero_point}")
```

---

## Advanced QAT Techniques

### Learnable FakeQuantize

Standard FakeQuantize uses fixed scale/zero_point computed from observers. Learnable FakeQuantize makes these parameters trainable.

```python
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize

class _LearnableFakeQuantize(FakeQuantizeBase):
    """
    Scale and zero_point are nn.Parameters, enabling gradient-based learning.

    Training modes:
    1. enable_static_estimate(): Use observer statistics
    2. enable_param_learning(): Learn scale/zp through backprop
    3. enable_static_observation(): Observe without updating qparams
    """

    def __init__(
        self,
        observer,
        quant_min=0,
        quant_max=255,
        scale=1.0,
        zero_point=0.0,
        channel_len=-1,          # -1 for per-tensor, >0 for per-channel
        use_grad_scaling=False,  # Normalize gradients by tensor size
        **observer_kwargs,
    ):
        # scale and zero_point are Parameters (trainable)
        self.scale = Parameter(torch.tensor([scale]))
        self.zero_point = Parameter(torch.tensor([zero_point]))
        ...
```

**Usage:**
```python
# Create learnable fake quantize
lfq = _LearnableFakeQuantize(
    observer=MinMaxObserver,
    quant_min=0,
    quant_max=255,
    use_grad_scaling=True,  # Recommended for stability
)

# Training phases:
# Phase 1: Warmup with observer (no fake quant)
lfq.enable_static_observation()
for batch in warmup_data:
    model(batch)

# Phase 2: Enable QAT with observer updates
lfq.enable_static_estimate()
for epoch in range(warmup_epochs):
    train_one_epoch(model)

# Phase 3: Learn scale/zero_point directly
lfq.enable_param_learning()
for epoch in range(finetune_epochs):
    train_one_epoch(model)
```

### Gradient Scaling

For learnable quantization, gradient scaling improves training stability:

```python
# From _LearnableFakeQuantize.forward():
if self.use_grad_scaling:
    grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
else:
    grad_factor = 1.0

# Gradients for scale/zero_point are scaled by grad_factor
X = torch._fake_quantize_learnable_per_tensor_affine(
    X, self.scale, self.zero_point,
    self.quant_min, self.quant_max, grad_factor
)
```

**Reference:** ["Learned Step Size Quantization"](https://openreview.net/pdf?id=rkgO66VKDS) (ICLR 2020)

---

### QAT Training Recipes

#### Recipe 1: Standard QAT

```python
import torch.ao.quantization as quant

model = MyModel()
model.train()

# Configure
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# Fuse modules first (important for accuracy)
model = quant.fuse_modules(model, [['conv', 'bn', 'relu']])

# Prepare
quant.prepare_qat(model, inplace=True)

# Train normally
for epoch in range(num_epochs):
    for batch, target in train_loader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Freeze BatchNorm after sufficient training
if epoch > freeze_bn_epoch:
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

# Convert
model.eval()
quantized = quant.convert(model.eval(), inplace=False)
```

#### Recipe 2: Progressive QAT

Start with float training, gradually introduce quantization:

```python
# Phase 1: Float training
model.apply(lambda m: m.disable_fake_quant() if hasattr(m, 'disable_fake_quant') else None)
train_epochs(model, num_float_epochs)

# Phase 2: Enable observers only
model.apply(lambda m: m.enable_observer() if hasattr(m, 'enable_observer') else None)
model.apply(lambda m: m.disable_fake_quant() if hasattr(m, 'disable_fake_quant') else None)
train_epochs(model, num_observer_epochs)

# Phase 3: Full QAT
model.apply(lambda m: m.enable_fake_quant() if hasattr(m, 'enable_fake_quant') else None)
train_epochs(model, num_qat_epochs)
```

#### Recipe 3: Low-Learning-Rate QAT

QAT often benefits from lower learning rates:

```python
# Reduce LR for QAT
qat_lr = float_lr * 0.1

# Use cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_qat_epochs
)

# Longer warmup
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, total_iters=1000
)
```

---

### Per-Channel vs Per-Tensor Quantization

#### Per-Tensor (Default for Activations)

Single scale/zero_point for entire tensor:
- Simpler, faster
- Lower accuracy
- Good for activations

```python
qconfig = QConfig(
    activation=MinMaxObserver.with_args(
        qscheme=torch.per_tensor_affine
    ),
    weight=...
)
```

#### Per-Channel (Recommended for Weights)

Different scale/zero_point per output channel:
- Better accuracy (especially for depthwise convs)
- Slightly more overhead
- Critical for weight quantization

```python
qconfig = QConfig(
    activation=...,
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0  # Output channel axis
    )
)
```

---

### Observer Comparison

| Observer | Use Case | Pros | Cons |
|----------|----------|------|------|
| `MinMaxObserver` | Simple calibration | Fast, simple | Sensitive to outliers |
| `MovingAverageMinMaxObserver` | Training/QAT | Smooth updates | Lag behind data |
| `HistogramObserver` | Optimal range | Handles outliers | Slower, more memory |
| `PerChannelMinMaxObserver` | Weight quantization | Per-channel accuracy | Per-channel overhead |
| `PlaceholderObserver` | Dynamic quantization | No calibration needed | Runtime overhead |

### HistogramObserver Details

Uses histogram to find optimal quantization range (minimizes quantization error):

```python
class HistogramObserver(UniformQuantizationObserverBase):
    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ):
        self.bins = bins
        self.upsample_rate = upsample_rate
        ...

    def _compute_quantization_error(self, scale, zero_point):
        """Compute MSE between original and quantized histogram."""
        # Used to find optimal scale/zero_point
```

---

### Handling Difficult Layers

#### Depthwise Separable Convolutions

Per-channel quantization is critical:

```python
# Standard conv: per-tensor weights OK
# Depthwise conv: MUST use per-channel weights

depthwise_qconfig = QConfig(
    activation=MinMaxObserver.with_args(dtype=torch.quint8),
    weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric
    )
)

# Apply specifically to depthwise layers
model.depthwise_conv.qconfig = depthwise_qconfig
```

#### Attention Mechanisms

Attention often has high dynamic range:

```python
# Use histogram observer for attention
attention_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=torch.quint8,
        bins=2048
    ),
    weight=default_per_channel_weight_observer
)

model.attention.qconfig = attention_qconfig
```

#### Skip Connections

Ensure compatible quantization at add/concat:

```python
class QuantizableResidualBlock(nn.Module):
    def __init__(self):
        self.add_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Quantized add handles scale matching
        out = self.add_relu.add_relu(out, identity)
        return out
```

---

### Module Fusion Details

Fusion combines consecutive operations for better quantization:

```python
from torch.ao.quantization import fuse_modules, fuse_modules_qat

# Common fusion patterns
fusion_patterns = [
    ['conv', 'bn'],           # Conv + BatchNorm
    ['conv', 'bn', 'relu'],   # Conv + BatchNorm + ReLU
    ['conv', 'relu'],         # Conv + ReLU
    ['linear', 'relu'],       # Linear + ReLU
    ['bn', 'relu'],           # BatchNorm + ReLU
]

# For eval mode (PTQ)
model.eval()
model = fuse_modules(model, [['conv1', 'bn1', 'relu1']])

# For train mode (QAT)
model.train()
model = fuse_modules_qat(model, [['conv1', 'bn1', 'relu1']])
```

**Why fusion helps:**
1. Eliminates intermediate quantize/dequantize
2. Folded BN doesn't add quantization error
3. ReLU can be fused without separate quantization

---

### Quantization-Aware Training Internals

#### FakeQuantize Forward Pass

```python
def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    """
    1. Quantize: x_q = clamp(round(x / scale) + zero_point, quant_min, quant_max)
    2. Dequantize: x_out = (x_q - zero_point) * scale

    Gradient: Straight-through estimator (STE)
    - Forward: applies quantization
    - Backward: gradient passes through unchanged (where not clamped)
    """
    x_q = torch.clamp(
        torch.round(x / scale) + zero_point,
        quant_min, quant_max
    )
    x_out = (x_q - zero_point) * scale
    return x_out
```

#### Straight-Through Estimator (STE)

```python
# STE gradient for quantization
class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        x_q = quantize(x, scale, zero_point)
        x_q = clamp(x_q, quant_min, quant_max)
        x_dq = dequantize(x_q, scale, zero_point)

        # Save for backward
        ctx.save_for_backward(x, scale)
        ctx.quant_min = quant_min
        ctx.quant_max = quant_max
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # Zero gradient where clamped (outside valid range)
        x_q = x / scale
        mask = (x_q >= ctx.quant_min) & (x_q <= ctx.quant_max)
        grad_input = grad_output * mask.float()
        return grad_input, None, None, None, None
```

---

### Debugging QAT

#### Visualize Quantization Ranges

```python
def plot_quantization_stats(model):
    import matplotlib.pyplot as plt

    scales = []
    names = []
    for name, module in model.named_modules():
        if hasattr(module, 'activation_post_process'):
            obs = module.activation_post_process
            if hasattr(obs, 'min_val') and hasattr(obs, 'max_val'):
                scale, zp = obs.calculate_qparams()
                scales.append(scale.item())
                names.append(name)

    plt.bar(range(len(scales)), scales)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel('Scale')
    plt.title('Quantization Scales by Layer')
    plt.tight_layout()
    plt.show()
```

#### Check for Outliers

```python
def check_activation_outliers(model, dataloader, threshold=6.0):
    """Check for activations that may cause quantization issues."""
    model.eval()
    outlier_layers = []

    hooks = []
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                mean = output.mean().item()
                std = output.std().item()
                max_val = output.abs().max().item()
                if max_val > mean + threshold * std:
                    outlier_layers.append((name, max_val, mean, std))
        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in dataloader:
            model(batch)
            break

    for hook in hooks:
        hook.remove()

    return outlier_layers
```

#### Compare Layer-by-Layer

```python
def compare_quantized_float(float_model, quant_model, input_data):
    """Compare intermediate activations between float and quantized models."""
    float_activations = {}
    quant_activations = {}

    def make_hook(storage, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                storage[name] = output.detach().clone()
        return hook

    # Register hooks
    for name, module in float_model.named_modules():
        module.register_forward_hook(make_hook(float_activations, name))
    for name, module in quant_model.named_modules():
        module.register_forward_hook(make_hook(quant_activations, name))

    # Run models
    float_model(input_data)
    quant_model(input_data)

    # Compare
    differences = {}
    for name in float_activations:
        if name in quant_activations:
            diff = (float_activations[name] - quant_activations[name]).abs()
            differences[name] = {
                'max': diff.max().item(),
                'mean': diff.mean().item(),
                'relative': (diff / (float_activations[name].abs() + 1e-8)).mean().item()
            }

    return differences
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| Observer | Collect statistics for quantization parameters |
| FakeQuantize | Simulate quantization during training |
| _LearnableFakeQuantize | Trainable scale/zero_point |
| QConfig | Configure observers for activations/weights |
| `prepare()` | Insert observers into model |
| `prepare_qat()` | Prepare model for QAT |
| `convert()` | Convert to quantized model |
| `quantize_dynamic()` | One-step dynamic quantization |
| `fuse_modules()` | Fuse operations for better quantization |
| `fuse_modules_qat()` | Fuse for QAT training |
| QuantStub/DeQuantStub | Mark quantization boundaries |
| HistogramObserver | Optimal range via histogram analysis |
| PerChannelMinMaxObserver | Per-channel weight quantization |
