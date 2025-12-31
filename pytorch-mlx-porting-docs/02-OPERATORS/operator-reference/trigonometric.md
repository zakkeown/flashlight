# Trigonometric Operators Reference

## Overview

Trigonometric operators compute standard mathematical functions on angles (in radians unless specified). These operations are fundamental for signal processing, geometric transformations, periodic phenomena, and many scientific computing tasks.

**Key Characteristics**:
- **Element-wise**: Operate independently on each element
- **Domain-specific**: Some have restricted input domains (e.g., asin requires [-1, 1])
- **Periodic**: sin, cos, tan have periodic behavior
- **Differentiable**: All trig functions have well-defined gradients
- **Pointwise tag**: Eligible for kernel fusion optimizations

**Common Applications**:
- **Signal processing**: Fourier transforms, wave generation
- **Positional encodings**: Transformer position embeddings
- **Physics simulations**: Oscillations, rotations, projectile motion
- **Computer vision**: Rotation matrices, homography transformations
- **Activation functions**: Periodic alternatives to ReLU

---

## Week 4 Day 1 Operators - Basic Trigonometric Functions

### sin

**Purpose**: Compute sine of input (element-wise)

**Signature**: `sin(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: sin(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  structured_delegate: sin.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = sin(self[i])
# Input assumed to be in radians
# Domain: (-∞, ∞)
# Range: [-1, 1]

result[i] = sin(self[i])
```

**Mathematical Properties**:
- **Period**: 2π
- **Symmetry**: Odd function, sin(-x) = -sin(x)
- **Special values**:
  - sin(0) = 0
  - sin(π/2) = 1
  - sin(π) = 0
  - sin(3π/2) = -1

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void sin_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "sin_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::sin(x);  // C++ standard library
        },
        [](Vectorized<scalar_t> x) {
          return x.sin();  // Vectorized SIMD operation
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor sin_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph sinWithTensor:selfTensor
                                              name:@"sin"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂sin(x)/∂x = cos(x)

Mathematical derivation:
Let y = sin(x)

dy/dx = lim[h→0] (sin(x+h) - sin(x))/h
      = lim[h→0] (sin(x)cos(h) + cos(x)sin(h) - sin(x))/h
      = lim[h→0] sin(x)(cos(h)-1)/h + cos(x)sin(h)/h
      = sin(x) · 0 + cos(x) · 1
      = cos(x)

Chain rule application:
If z = sin(f(x)), then:
∂z/∂x = cos(f(x)) · ∂f/∂x
```

**MLX Equivalent**:
```python
import mlx.core as mx

def sin(x):
    """
    PyTorch: torch.sin(x)
    MLX: mx.sin(x)
    """
    return mx.sin(x)

# Example 1: Basic sine computation
angles = mx.array([0, mx.pi/2, mx.pi, 3*mx.pi/2, 2*mx.pi])
result = mx.sin(angles)
# [0.0, 1.0, 0.0 (≈1e-16), -1.0, 0.0 (≈-2e-16)]

# Example 2: Wave generation
t = mx.linspace(0, 2*mx.pi, 100)
frequency = 5.0
amplitude = 2.0
wave = amplitude * mx.sin(frequency * t)

# Example 3: Transformer positional encoding
def get_positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding for Transformers"""
    position = mx.arange(seq_len)[:, None]  # [seq_len, 1]
    div_term = mx.exp(mx.arange(0, d_model, 2) * 
                      -(mx.log(10000.0) / d_model))
    
    pe = mx.zeros((seq_len, d_model))
    pe[:, 0::2] = mx.sin(position * div_term)  # Even indices
    pe[:, 1::2] = mx.cos(position * div_term)  # Odd indices
    return pe

# Example 4: Rotation matrix (2D)
theta = mx.array([mx.pi/4])  # 45 degrees
rotation = mx.array([
    [mx.cos(theta), -mx.sin(theta)],
    [mx.sin(theta),  mx.cos(theta)]
])
```

**Common Patterns**:
```python
# Pattern 1: Positional encoding (Transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0)]

# Pattern 2: Periodic activation function
class SineActivation(nn.Module):
    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = omega
    
    def forward(self, x):
        return torch.sin(self.omega * x)

# Pattern 3: Fourier features for coordinate-based MLPs
def fourier_features(coords, num_freqs):
    """
    coords: [batch, spatial_dims]
    Returns: [batch, spatial_dims * num_freqs * 2]
    """
    freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
    features = []
    for freq in freq_bands:
        features.append(torch.sin(2 * math.pi * coords * freq))
        features.append(torch.cos(2 * math.pi * coords * freq))
    return torch.cat(features, dim=-1)

# Pattern 4: Wave synthesis
def synthesize_wave(frequencies, amplitudes, phases, sample_rate, duration):
    """Generate complex waveform from multiple frequencies"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.zeros_like(t)
    
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        waveform += amp * torch.sin(2 * math.pi * freq * t + phase)
    
    return waveform
```

**Edge Cases**:
- **Very large inputs**: Accuracy degrades for |x| > 1e8 due to period reduction
- **Numerical precision**: sin(π) ≈ 1.2e-16 (not exactly 0) due to π approximation
- **Complex inputs**: Supported, uses complex sine formula
- **NaN/Inf**: sin(NaN) = NaN, sin(±Inf) = NaN

**Performance Notes**:
- CPU: Uses vectorized SIMD instructions (SSE, AVX)
- MPS: Parallel GPU execution
- Taylor series approximation for small inputs
- Range reduction for large inputs (x mod 2π)
- Pointwise tag enables kernel fusion

**MLX Porting Considerations**:
- Direct equivalent: `mx.sin()`
- Identical semantics
- Same numerical precision characteristics
- Complex number support may differ

---

### cos

**Purpose**: Compute cosine of input (element-wise)

**Signature**: `cos(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: cos(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  structured_delegate: cos.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = cos(self[i])
# Input assumed to be in radians
# Domain: (-∞, ∞)
# Range: [-1, 1]

result[i] = cos(self[i])
```

**Mathematical Properties**:
- **Period**: 2π
- **Symmetry**: Even function, cos(-x) = cos(x)
- **Special values**:
  - cos(0) = 1
  - cos(π/2) = 0
  - cos(π) = -1
  - cos(3π/2) = 0

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void cos_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "cos_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::cos(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.cos();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor cos_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph cosWithTensor:selfTensor
                                              name:@"cos"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂cos(x)/∂x = -sin(x)

Mathematical derivation:
Let y = cos(x)

dy/dx = lim[h→0] (cos(x+h) - cos(x))/h
      = lim[h→0] (cos(x)cos(h) - sin(x)sin(h) - cos(x))/h
      = lim[h→0] cos(x)(cos(h)-1)/h - sin(x)sin(h)/h
      = cos(x) · 0 - sin(x) · 1
      = -sin(x)

Chain rule application:
If z = cos(f(x)), then:
∂z/∂x = -sin(f(x)) · ∂f/∂x
```

**MLX Equivalent**:
```python
import mlx.core as mx

def cos(x):
    """
    PyTorch: torch.cos(x)
    MLX: mx.cos(x)
    """
    return mx.cos(x)

# Example 1: Basic cosine computation
angles = mx.array([0, mx.pi/2, mx.pi, 3*mx.pi/2, 2*mx.pi])
result = mx.cos(angles)
# [1.0, 0.0 (≈6e-17), -1.0, 0.0 (≈-1.8e-16), 1.0]

# Example 2: Rotation matrix (2D)
theta = mx.array([mx.pi/3])  # 60 degrees
cos_theta = mx.cos(theta)
sin_theta = mx.sin(theta)
rotation_matrix = mx.array([
    [cos_theta, -sin_theta],
    [sin_theta,  cos_theta]
])

# Example 3: Attention mask based on distance
def positional_decay_mask(seq_len, decay_rate=0.1):
    """Cosine-based attention decay with distance"""
    positions = mx.arange(seq_len)[:, None] - mx.arange(seq_len)[None, :]
    distances = mx.abs(positions)
    
    # Cosine decay: 1 at distance 0, decreases with distance
    mask = mx.cos(mx.pi * distances * decay_rate / seq_len)
    mask = mx.maximum(mask, 0.0)  # Clip negative values
    return mask

# Example 4: Learnable frequency (implicit neural representations)
class SirenLayer(nn.Module):
    """SIREN: Sinusoidal Representation Networks"""
    def __init__(self, in_features, out_features, omega_0=30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
    
    def forward(self, x):
        # First layer uses sine, but cosine also used in some variants
        return torch.sin(self.omega_0 * self.linear(x))
```

**Common Patterns**:
```python
# Pattern 1: Positional encoding (complementary to sin)
class PositionalEncoding(nn.Module):
    def forward(self, x):
        position = torch.arange(len(x)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(len(x), d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)  # Orthogonal to sin
        return x + pe

# Pattern 2: Cosine similarity normalization
def cosine_similarity_attention(query, key):
    """
    Attention using cosine similarity instead of dot product
    """
    # Normalize
    query_norm = query / torch.norm(query, dim=-1, keepdim=True)
    key_norm = key / torch.norm(key, dim=-1, keepdim=True)
    
    # Cosine similarity via dot product of normalized vectors
    similarity = query_norm @ key_norm.transpose(-2, -1)
    return similarity

# Pattern 3: Cosine learning rate schedule
def cosine_annealing_lr(step, total_steps, max_lr, min_lr=0):
    """Cosine annealing learning rate schedule"""
    progress = step / total_steps
    lr = min_lr + (max_lr - min_lr) * 0.5 * (
        1 + torch.cos(torch.tensor(math.pi * progress))
    )
    return lr.item()

# Pattern 4: Direction vector from angles
def spherical_to_cartesian(theta, phi):
    """
    Convert spherical coordinates to Cartesian
    theta: azimuthal angle [0, 2π]
    phi: polar angle [0, π]
    Returns: [x, y, z] unit vector
    """
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)
```

**Edge Cases**:
- **Very large inputs**: Accuracy degrades for |x| > 1e8
- **Numerical precision**: cos(π/2) ≈ 6e-17 (not exactly 0)
- **Complex inputs**: Supported
- **NaN/Inf**: cos(NaN) = NaN, cos(±Inf) = NaN

**Relationship to sin**:
```python
# Trigonometric identities
torch.cos(x) == torch.sin(x + math.pi/2)  # Phase shift
torch.sin(x)**2 + torch.cos(x)**2 == 1  # Pythagorean identity
torch.cos(2*x) == torch.cos(x)**2 - torch.sin(x)**2  # Double angle
```

**Performance Notes**:
- Same performance characteristics as sin
- Often computed together with sin for efficiency
- Vectorized SIMD operations
- Kernel fusion eligible

**MLX Porting Considerations**:
- Direct equivalent: `mx.cos()`
- Identical semantics
- Same numerical precision

---

### tan

**Purpose**: Compute tangent of input (element-wise)

**Signature**: `tan(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml):
```yaml
- func: tan(Tensor self) -> Tensor
  variants: function, method
  device_check: NoCheck
  structured_delegate: tan.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = tan(self[i])
# Defined as tan(x) = sin(x) / cos(x)
# Input assumed to be in radians
# Domain: x ≠ π/2 + nπ (where cos(x) = 0)
# Range: (-∞, ∞)

result[i] = tan(self[i]) = sin(self[i]) / cos(self[i])
```

**Mathematical Properties**:
- **Period**: π (not 2π like sin/cos)
- **Symmetry**: Odd function, tan(-x) = -tan(x)
- **Singularities**: Vertical asymptotes at x = π/2 + nπ
- **Special values**:
  - tan(0) = 0
  - tan(π/4) = 1
  - tan(π/2) → ±∞ (undefined)
  - tan(-π/4) = -1

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void tan_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "tan_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::tan(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.tan();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor tan_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph tanWithTensor:selfTensor
                                              name:@"tan"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂tan(x)/∂x = sec²(x) = 1/cos²(x) = 1 + tan²(x)

Mathematical derivation:
Let y = tan(x) = sin(x)/cos(x)

Using quotient rule:
dy/dx = (cos(x)·cos(x) - sin(x)·(-sin(x))) / cos²(x)
      = (cos²(x) + sin²(x)) / cos²(x)
      = 1 / cos²(x)
      = sec²(x)

Alternative form:
dy/dx = 1 + tan²(x)  (since 1 + tan²(x) = sec²(x))

Chain rule application:
If z = tan(f(x)), then:
∂z/∂x = sec²(f(x)) · ∂f/∂x
      = (1 + tan²(f(x))) · ∂f/∂x
```

**Backward Implementation**:
```cpp
Tensor tan_backward(const Tensor& grad, const Tensor& input, const Tensor& output) {
  // ∂tan/∂x = 1 / cos²(x) = 1 + tan²(x)
  // Using output (already computed tan) is more efficient
  return grad * (1 + output.pow(2));
  
  // Alternative (less efficient):
  // auto cos_x = input.cos();
  // return grad / (cos_x * cos_x);
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def tan(x):
    """
    PyTorch: torch.tan(x)
    MLX: mx.tan(x)
    """
    return mx.tan(x)

# Example 1: Basic tangent computation
angles = mx.array([0, mx.pi/4, mx.pi/3, -mx.pi/4])
result = mx.tan(angles)
# [0.0, 1.0, 1.732 (≈√3), -1.0]

# Example 2: Slope from angle
def angle_to_slope(theta):
    """
    Convert angle (in radians) to slope
    theta: angle from horizontal
    Returns: dy/dx slope
    """
    return mx.tan(theta)

# 45° angle has slope 1
slope_45 = angle_to_slope(mx.array([mx.pi/4]))  # 1.0

# Example 3: Tangent activation (less common than tanh)
class TanActivation(nn.Module):
    """Tangent activation (rarely used, unbounded)"""
    def forward(self, x):
        # Scale input to avoid asymptotes
        return torch.tan(torch.clamp(x, -1.5, 1.5))

# Example 4: Perspective projection
def perspective_fov(fov_degrees, aspect_ratio, near, far):
    """
    Compute perspective projection matrix
    fov_degrees: vertical field of view in degrees
    """
    fov_rad = mx.radians(fov_degrees)
    f = 1.0 / mx.tan(fov_rad / 2.0)  # Focal length
    
    # Perspective matrix (simplified)
    return mx.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), -1],
        [0, 0, (2 * far * near) / (near - far), 0]
    ])
```

**Common Patterns**:
```python
# Pattern 1: Avoid asymptotes with clamping
def safe_tan(x, clamp_val=1.4):
    """Compute tan while avoiding asymptotes at ±π/2"""
    # π/2 ≈ 1.5708, so clamp to ±1.4 to stay safe
    x_safe = torch.clamp(x, -clamp_val, clamp_val)
    return torch.tan(x_safe)

# Pattern 2: Linear slope from angle
def compute_ray_direction(horizontal_angle, vertical_angle):
    """
    Compute 3D ray direction from spherical angles
    Returns normalized direction vector
    """
    # Horizontal plane
    dx = torch.cos(horizontal_angle)
    dy = torch.sin(horizontal_angle)
    
    # Vertical component
    dz = torch.tan(vertical_angle) * torch.sqrt(dx**2 + dy**2)
    
    # Normalize
    direction = torch.stack([dx, dy, dz], dim=-1)
    return direction / torch.norm(direction, dim=-1, keepdim=True)

# Pattern 3: Focal length computation (computer vision)
def focal_length_from_fov(fov_degrees, image_width):
    """
    Compute focal length in pixels from field of view
    fov_degrees: horizontal field of view
    image_width: image width in pixels
    """
    fov_rad = math.radians(fov_degrees)
    focal_length = image_width / (2 * torch.tan(torch.tensor(fov_rad / 2)))
    return focal_length

# Pattern 4: Gradient computation (rarely used as activation)
# Note: tan has unbounded derivative, making it unstable for deep learning
# tanh (hyperbolic tangent) is preferred for activations
```

**Edge Cases**:
- **Asymptotes**: tan(π/2 + nπ) → ±∞ (implementation returns large finite value)
- **Near asymptotes**: Numerical instability for x close to π/2 + nπ
- **Very large inputs**: Accuracy degrades due to period reduction
- **Complex inputs**: Supported
- **NaN/Inf**: tan(NaN) = NaN, tan(±Inf) = NaN
- **Output overflow**: Can produce very large values near asymptotes

**Comparison to tanh** (hyperbolic tangent):
```python
# tan: periodic, unbounded, asymptotes
torch.tan(x)  # Range: (-∞, ∞), period: π

# tanh: non-periodic, bounded, smooth
torch.tanh(x)  # Range: (-1, 1), no asymptotes

# tanh is much more commonly used in deep learning
# tan is used in geometric/physics computations
```

**Numerical Stability**:
- **Problem**: Division by cos(x) near asymptotes
- **Mitigation**: Implementations use range reduction and special handling
- **Recommendation**: Clamp inputs to avoid ±π/2 when using in gradients

**Performance Notes**:
- More expensive than sin/cos (division involved)
- Vectorized SIMD when available
- Special handling for asymptote regions
- Pointwise operation eligible for fusion

**MLX Porting Considerations**:
- Direct equivalent: `mx.tan()`
- Identical semantics
- Same asymptote behavior
- Numerical stability similar

---

## Summary - Week 4 Day 1

**Total Operators Documented**: 3 basic trigonometric functions

**Day 1 - Basic Trig** (3 ops):
- `sin`, `cos`, `tan`

**Mathematical Properties Summary**:

| Function | Domain | Range | Period | Symmetry | Derivative |
|----------|--------|-------|--------|----------|------------|
| sin(x) | ℝ | [-1, 1] | 2π | Odd | cos(x) |
| cos(x) | ℝ | [-1, 1] | 2π | Even | -sin(x) |
| tan(x) | x ≠ π/2+nπ | ℝ | π | Odd | sec²(x) |

**Common Applications**:
- **Positional Encodings**: Transformer position embeddings (sin/cos pairs)
- **Rotation Matrices**: 2D/3D geometric transformations
- **Wave Synthesis**: Signal processing, audio generation
- **Periodic Activations**: SIREN and Fourier feature networks
- **Learning Rate Schedules**: Cosine annealing
- **Perspective Projection**: Computer vision (tan for FOV)

**Numerical Considerations**:
- **Large inputs**: Precision loss for |x| > 1e8 due to period reduction
- **Asymptotes**: tan(x) unstable near π/2 + nπ
- **Special values**: π approximation causes small errors (sin(π) ≈ 1e-16, not 0)

**PyTorch → MLX Mapping**:
- `torch.sin` → `mx.sin`
- `torch.cos` → `mx.cos`
- `torch.tan` → `mx.tan`
- All have identical semantics

**Gradient Formulas**:
- ∂sin(x)/∂x = cos(x)
- ∂cos(x)/∂x = -sin(x)
- ∂tan(x)/∂x = sec²(x) = 1 + tan²(x)

**Progress**: 3 / 15 trigonometric operators documented (20%)
**Week 4 Day 1**: sin, cos, tan ✅
**Week 4 Day 2**: asin, acos, atan, atan2 (pending)
**Week 4 Day 3**: sinh, cosh, tanh (pending)
**Week 4 Day 4**: asinh, acosh, atanh (pending)
**Week 4 Day 5**: deg2rad, rad2deg (pending)


---

## Week 4 Day 2 Operators - Inverse Trigonometric Functions

### asin

**Purpose**: Compute arcsine (inverse sine) of input (element-wise)

**Signature**: `asin(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:977-984):
```yaml
- func: asin(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: asin.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asin_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asin_sparse_csr
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = arcsin(self[i])
# Returns angle in radians whose sine is self[i]
# Domain: [-1, 1]
# Range: [-π/2, π/2]

result[i] = arcsin(self[i])
```

**Mathematical Properties**:
- **Inverse of sin**: asin(sin(x)) = x for x ∈ [-π/2, π/2]
- **Restricted domain**: Input must be in [-1, 1]
- **Range**: Output in [-π/2, π/2] radians
- **Symmetry**: Odd function, asin(-x) = -asin(x)
- **Special values**:
  - asin(0) = 0
  - asin(1) = π/2
  - asin(-1) = -π/2
  - asin(√2/2) = π/4

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void asin_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "asin_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::asin(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.asin();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor asin_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph asinWithTensor:selfTensor
                                               name:@"asin"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂asin(x)/∂x = 1 / √(1 - x²)

Mathematical derivation:
Let y = asin(x), which means sin(y) = x

Differentiate both sides with respect to x:
cos(y) · dy/dx = 1
dy/dx = 1 / cos(y)

From sin²(y) + cos²(y) = 1:
cos(y) = √(1 - sin²(y)) = √(1 - x²)

Therefore:
dy/dx = 1 / √(1 - x²)

Chain rule application:
If z = asin(f(x)), then:
∂z/∂x = (1 / √(1 - f(x)²)) · ∂f/∂x

Note: Gradient becomes infinite as x → ±1
```

**Backward Implementation**:
```cpp
Tensor asin_backward(const Tensor& grad, const Tensor& input) {
  // ∂asin/∂x = 1 / sqrt(1 - x²)
  auto one_minus_x_sq = 1 - input.pow(2);
  return grad / one_minus_x_sq.sqrt();
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def asin(x):
    """
    PyTorch: torch.asin(x)
    MLX: mx.arcsin(x)
    
    Note: MLX uses 'arcsin' naming (NumPy convention)
    """
    return mx.arcsin(x)

# Example 1: Basic arcsine computation
values = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0])
angles = mx.arcsin(values)
# [-π/2, -π/6, 0, π/6, π/2]
# [-1.571, -0.524, 0.0, 0.524, 1.571]

# Example 2: Recover angle from sine
sin_theta = mx.array([0.5])  # sin(30°) = 0.5
theta = mx.arcsin(sin_theta)  # π/6 radians = 30°
print(mx.degrees(theta))  # 30.0

# Example 3: Angle from y-coordinate on unit circle
def unit_circle_angle_from_y(y):
    """
    Given y-coordinate on unit circle, find angle
    Assumes y ∈ [-1, 1]
    Returns angle in radians from -π/2 to π/2
    """
    return mx.arcsin(mx.clip(y, -1.0, 1.0))

# Example 4: Inverse of positional encoding component
def recover_position_from_encoding(encoding, d_model):
    """
    Attempt to recover position from sinusoidal encoding
    (Simplified, works for first component)
    """
    # First encoding component: sin(pos / 10000^(0/d_model))
    angle = mx.arcsin(encoding[:, 0])
    position = angle  # First term has div_term ≈ 1
    return position
```

**Common Patterns**:
```python
# Pattern 1: Angle from normalized coordinate
def coord_to_angle(normalized_coord):
    """
    Convert normalized coordinate [-1, 1] to angle
    Used in spherical projections
    """
    # Clamp to avoid domain errors from numerical precision
    safe_coord = torch.clamp(normalized_coord, -1.0, 1.0)
    return torch.asin(safe_coord)

# Pattern 2: Elevation angle from height
def height_to_elevation_angle(height, radius):
    """
    height: vertical position
    radius: distance from origin
    Returns elevation angle from horizontal plane
    """
    sin_elevation = height / radius
    sin_elevation = torch.clamp(sin_elevation, -1.0, 1.0)
    return torch.asin(sin_elevation)

# Pattern 3: Inverse spherical coordinate transformation
def cartesian_to_spherical_phi(x, y, z):
    """
    Extract polar angle φ (from z-axis) from Cartesian coordinates
    """
    r = torch.sqrt(x**2 + y**2 + z**2)
    # cos(φ) = z/r, so φ = acos(z/r)
    # But can also use: sin(φ) = sqrt(x²+y²)/r
    sin_phi = torch.sqrt(x**2 + y**2) / r
    return torch.asin(torch.clamp(sin_phi, -1.0, 1.0))

# Pattern 4: Safe domain clamping (critical!)
def safe_asin(x, epsilon=1e-6):
    """
    Compute asin with safety margin to avoid domain errors
    """
    # Clamp to slightly inside [-1, 1] to account for numerical errors
    x_safe = torch.clamp(x, -1.0 + epsilon, 1.0 - epsilon)
    return torch.asin(x_safe)
```

**Edge Cases**:
- **Domain violation**: asin(x) for |x| > 1 returns NaN
- **Boundary values**: asin(±1) = ±π/2 (well-defined)
- **Gradient at boundaries**: → ∞ as x → ±1 (numerical instability)
- **Complex inputs**: Supported for |x| > 1 via complex extension
- **NaN propagation**: asin(NaN) = NaN

**Numerical Stability**:
- **Problem**: Gradient 1/√(1-x²) explodes as x → ±1
- **Mitigation**: Clamp inputs to [-1+ε, 1-ε] when possible
- **Common error**: Small numerical errors can push x slightly outside [-1, 1]
- **Recommendation**: Always clamp inputs before asin in production code

**Performance Notes**:
- More expensive than sin (iterative or rational approximation)
- Vectorized SIMD when available
- Pointwise operation eligible for fusion
- Sparse tensor support

**MLX Porting Considerations**:
- MLX uses `arcsin` (not `asin`) following NumPy convention
- Identical semantics and domain restrictions
- Same numerical stability considerations

---

### acos

**Purpose**: Compute arccosine (inverse cosine) of input (element-wise)

**Signature**: `acos(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:514-518):
```yaml
- func: acos(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: acos.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = arccos(self[i])
# Returns angle in radians whose cosine is self[i]
# Domain: [-1, 1]
# Range: [0, π]

result[i] = arccos(self[i])
```

**Mathematical Properties**:
- **Inverse of cos**: acos(cos(x)) = x for x ∈ [0, π]
- **Restricted domain**: Input must be in [-1, 1]
- **Range**: Output in [0, π] radians (always positive!)
- **Symmetry**: acos(-x) = π - acos(x)
- **Special values**:
  - acos(1) = 0
  - acos(0) = π/2
  - acos(-1) = π
  - acos(√2/2) = π/4

**Relationship to asin**:
```python
acos(x) = π/2 - asin(x)  # Complementary angles
```

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void acos_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "acos_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::acos(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.acos();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor acos_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph acosWithTensor:selfTensor
                                               name:@"acos"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂acos(x)/∂x = -1 / √(1 - x²)

Mathematical derivation:
Let y = acos(x), which means cos(y) = x

Differentiate both sides:
-sin(y) · dy/dx = 1
dy/dx = -1 / sin(y)

From sin²(y) + cos²(y) = 1:
sin(y) = √(1 - cos²(y)) = √(1 - x²)

Therefore:
dy/dx = -1 / √(1 - x²)

Alternative derivation from acos(x) = π/2 - asin(x):
d/dx[acos(x)] = d/dx[π/2 - asin(x)]
              = 0 - 1/√(1-x²)
              = -1/√(1-x²)

Chain rule:
If z = acos(f(x)), then:
∂z/∂x = (-1 / √(1 - f(x)²)) · ∂f/∂x
```

**Backward Implementation**:
```cpp
Tensor acos_backward(const Tensor& grad, const Tensor& input) {
  // ∂acos/∂x = -1 / sqrt(1 - x²)
  auto one_minus_x_sq = 1 - input.pow(2);
  return -grad / one_minus_x_sq.sqrt();
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def acos(x):
    """
    PyTorch: torch.acos(x)
    MLX: mx.arccos(x)
    """
    return mx.arccos(x)

# Example 1: Basic arccosine computation
values = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0])
angles = mx.arccos(values)
# [π, 2π/3, π/2, π/3, 0]
# [3.142, 2.094, 1.571, 1.047, 0.0]

# Example 2: Angle between two vectors
def angle_between_vectors(v1, v2):
    """
    Compute angle between two vectors using dot product
    angle = acos(v1·v2 / (|v1||v2|))
    """
    dot_product = mx.sum(v1 * v2)
    norm_product = mx.linalg.norm(v1) * mx.linalg.norm(v2)
    cos_angle = dot_product / norm_product
    
    # Clamp to avoid domain errors from numerical precision
    cos_angle = mx.clip(cos_angle, -1.0, 1.0)
    return mx.arccos(cos_angle)

v1 = mx.array([1.0, 0.0, 0.0])
v2 = mx.array([0.0, 1.0, 0.0])
angle = angle_between_vectors(v1, v2)  # π/2 (90 degrees)

# Example 3: Latitude from z-coordinate on sphere
def z_to_latitude(z, radius=1.0):
    """
    Convert z-coordinate to latitude angle
    z/radius = cos(latitude_from_pole)
    """
    cos_lat = z / radius
    cos_lat = mx.clip(cos_lat, -1.0, 1.0)
    # This gives angle from north pole, convert to latitude
    pole_angle = mx.arccos(cos_lat)
    latitude = mx.pi/2 - pole_angle
    return latitude

# Example 4: Reflection angle (Fresnel equations)
def fresnel_reflection_coefficient(n1, n2, incident_angle):
    """
    Compute Fresnel reflection coefficient
    n1, n2: refractive indices
    incident_angle: angle of incidence in radians
    """
    # Snell's law: n1*sin(θ1) = n2*sin(θ2)
    sin_transmitted = (n1 / n2) * mx.sin(incident_angle)
    sin_transmitted = mx.clip(sin_transmitted, -1.0, 1.0)
    
    # Transmitted angle
    cos_transmitted_sq = 1 - sin_transmitted**2
    if mx.any(cos_transmitted_sq < 0):
        # Total internal reflection
        return 1.0
    
    transmitted_angle = mx.arccos(mx.sqrt(cos_transmitted_sq))
    return transmitted_angle
```

**Common Patterns**:
```python
# Pattern 1: Angle between vectors (most common use)
class CosineSimilarity(nn.Module):
    def forward(self, x1, x2):
        # Cosine similarity
        cos_sim = F.cosine_similarity(x1, x2, dim=-1)
        
        # Convert to angle (optional, for interpretation)
        cos_sim_clamped = torch.clamp(cos_sim, -1.0, 1.0)
        angle = torch.acos(cos_sim_clamped)
        return angle

# Pattern 2: Spherical coordinate conversion
def cartesian_to_spherical_theta(x, y, z):
    """
    Polar angle θ (from z-axis) in spherical coordinates
    θ = acos(z / r)
    """
    r = torch.sqrt(x**2 + y**2 + z**2)
    cos_theta = z / (r + 1e-8)  # Add epsilon to avoid division by zero
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.acos(cos_theta)

# Pattern 3: Safe domain handling (critical!)
def safe_acos(x, epsilon=1e-6):
    """Compute acos with safety clamping"""
    x_safe = torch.clamp(x, -1.0 + epsilon, 1.0 - epsilon)
    return torch.acos(x_safe)

# Pattern 4: Batch vector angle computation
def batch_vector_angles(vectors1, vectors2):
    """
    Compute angles between corresponding vectors in batches
    vectors1, vectors2: [batch_size, vector_dim]
    Returns: [batch_size] angles in radians
    """
    # Normalize
    v1_norm = vectors1 / torch.norm(vectors1, dim=1, keepdim=True)
    v2_norm = vectors2 / torch.norm(vectors2, dim=1, keepdim=True)
    
    # Dot product
    cos_angles = torch.sum(v1_norm * v2_norm, dim=1)
    
    # Clamp and compute angle
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    return torch.acos(cos_angles)
```

**Edge Cases**:
- **Domain violation**: acos(x) for |x| > 1 returns NaN
- **Range**: Always returns [0, π], never negative
- **Gradient at boundaries**: → ∞ as x → ±1
- **Complex extension**: Supported for |x| > 1
- **Numerical precision**: cos(acos(x)) may not exactly equal x

**Comparison to asin**:
```python
# Range difference:
asin: [-π/2, π/2]  # Can be negative
acos: [0, π]        # Always positive

# Relationship:
acos(x) = π/2 - asin(x)

# Gradient sign:
∂asin/∂x = +1/√(1-x²)  # Positive
∂acos/∂x = -1/√(1-x²)  # Negative
```

**Performance Notes**:
- Similar cost to asin
- Often implemented as π/2 - asin(x)
- Vectorized SIMD operations
- Pointwise operation eligible for fusion

**MLX Porting Considerations**:
- MLX uses `arccos` (not `acos`)
- Identical semantics
- Same domain restrictions and numerical considerations

---

### atan

**Purpose**: Compute arctangent (inverse tangent) of input (element-wise)

**Signature**: `atan(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:1014-1021):
```yaml
- func: atan(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: atan.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atan_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atan_sparse_csr
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = arctan(self[i])
# Returns angle in radians whose tangent is self[i]
# Domain: (-∞, ∞)  # No restrictions!
# Range: (-π/2, π/2)

result[i] = arctan(self[i])
```

**Mathematical Properties**:
- **Inverse of tan**: atan(tan(x)) = x for x ∈ (-π/2, π/2)
- **Unrestricted domain**: Works for all real values
- **Range**: Output in (-π/2, π/2) radians
- **Symmetry**: Odd function, atan(-x) = -atan(x)
- **Asymptotes**: Horizontal asymptotes at ±π/2
- **Special values**:
  - atan(0) = 0
  - atan(1) = π/4
  - atan(-1) = -π/4
  - atan(∞) = π/2
  - atan(-∞) = -π/2

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void atan_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "atan_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::atan(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.atan();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor atan_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph atanWithTensor:selfTensor
                                               name:@"atan"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂atan(x)/∂x = 1 / (1 + x²)

Mathematical derivation:
Let y = atan(x), which means tan(y) = x

Differentiate both sides:
sec²(y) · dy/dx = 1
dy/dx = 1 / sec²(y)

Since sec²(y) = 1 + tan²(y) = 1 + x²:
dy/dx = 1 / (1 + x²)

Chain rule:
If z = atan(f(x)), then:
∂z/∂x = (1 / (1 + f(x)²)) · ∂f/∂x

Note: Gradient is always well-defined and bounded!
```

**Backward Implementation**:
```cpp
Tensor atan_backward(const Tensor& grad, const Tensor& input) {
  // ∂atan/∂x = 1 / (1 + x²)
  return grad / (1 + input.pow(2));
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def atan(x):
    """
    PyTorch: torch.atan(x)
    MLX: mx.arctan(x)
    """
    return mx.arctan(x)

# Example 1: Basic arctangent computation
values = mx.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
angles = mx.arctan(values)
# [-1.570 (≈-π/2), -π/4, 0, π/4, 1.570 (≈π/2)]

# Example 2: Slope to angle conversion
slopes = mx.array([0.0, 1.0, -1.0, 100.0])
angles_rad = mx.arctan(slopes)
angles_deg = mx.degrees(angles_rad)
# [0°, 45°, -45°, 89.4°]

# Example 3: Smooth thresholding (activation-like)
def soft_threshold(x, scale=1.0):
    """
    Smooth S-shaped function using atan
    Output bounded in (-π/2, π/2)
    """
    return mx.arctan(x / scale)

# Steeper than tanh, but unbounded gradient
x = mx.linspace(-10, 10, 100)
y = soft_threshold(x, scale=2.0)

# Example 4: Angle from gradient
def gradient_to_direction(dx, dy):
    """
    Convert gradient vector to angle
    Returns angle from horizontal axis
    Note: Limited to (-π/2, π/2), use atan2 for full circle
    """
    slope = dy / (dx + 1e-8)
    return mx.arctan(slope)
```

**Common Patterns**:
```python
# Pattern 1: Simple angle from slope (limited range!)
def slope_to_angle_simple(rise, run):
    """
    Compute angle from rise/run
    WARNING: Only gives angles in (-90°, 90°)
    For full circle, use atan2 instead
    """
    slope = rise / (run + 1e-8)
    return torch.atan(slope)

# Pattern 2: Smooth activation function
class AtanActivation(nn.Module):
    """
    Arctangent activation function
    - Bounded output: (-π/2, π/2)
    - Non-zero gradient everywhere
    - Smoother than ReLU, less saturating than tanh
    """
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return torch.atan(x / self.scale)

# Pattern 3: Normalized angle (rarely used alone, atan2 preferred)
def normalize_to_angle(x):
    """Map unbounded input to (-π/2, π/2)"""
    return torch.atan(x)

# Pattern 4: Gradient magnitude analysis
def gradient_magnitude_to_angle(grad_x, grad_y):
    """
    Convert gradient components to angle
    Note: This gives angle of gradient direction,
    but limited to (-π/2, π/2). Use atan2 for full range.
    """
    # This is incomplete! Use atan2 instead.
    # Shown here to illustrate limitation
    ratio = grad_y / (grad_x.abs() + 1e-8)
    return torch.atan(ratio)  # INCOMPLETE
```

**Edge Cases**:
- **No domain restrictions**: Works for all finite inputs
- **Infinity**: atan(±∞) = ±π/2
- **Large values**: Asymptotically approaches ±π/2
- **NaN**: atan(NaN) = NaN
- **Gradient**: Always well-defined, never infinite
- **Range limitation**: Cannot distinguish between quadrants (use atan2)

**Comparison to atan2**:
```python
# atan: Single argument, range (-π/2, π/2)
torch.atan(y/x)  # Limited range, ambiguous quadrants

# atan2: Two arguments, full range (-π, π]
torch.atan2(y, x)  # Full circle, unambiguous quadrants

# Example showing difference:
y, x = 1.0, 1.0   # First quadrant
atan(y/x) = π/4
atan2(y, x) = π/4  # Same

y, x = 1.0, -1.0  # Second quadrant
atan(y/x) = atan(-1) = -π/4  # WRONG quadrant!
atan2(y, x) = 3π/4  # Correct

# Recommendation: Use atan2 for angle computations
```

**Numerical Stability**:
- **Excellent**: No domain restrictions, bounded gradients
- **No clamping needed**: Unlike asin/acos
- **Asymptotic behavior**: Smooth saturation at extremes
- **Preferred over asin/acos**: When computing angles from ratios

**Performance Notes**:
- Less expensive than asin/acos (no domain checks)
- Rational approximation for intermediate values
- Asymptotic formulas for large inputs
- Vectorized SIMD operations
- Sparse tensor support

**MLX Porting Considerations**:
- MLX uses `arctan` (not `atan`)
- Identical semantics
- No domain restrictions to worry about

---

### atan2

**Purpose**: Compute two-argument arctangent (full-circle angle)

**Signature**: `atan2(Tensor self, Tensor other) -> Tensor`

**YAML Definition** (native_functions.yaml:9966-9970):
```yaml
- func: atan2(Tensor self, Tensor other) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: atan2.out
  variants: method, function
  tags: [core, pointwise]
# arctan2, alias of atan2
```

**Algorithm**:
```python
# Element-wise: result[i] = atan2(self[i], other[i])
# Returns angle from positive x-axis to point (other[i], self[i])
# 
# Signature: atan2(y, x) where:
#   - y = self (numerator, vertical component)
#   - x = other (denominator, horizontal component)
#
# Domain: All (y, x) except (0, 0)
# Range: (-π, π]

def atan2(y, x):
    if x > 0:
        return atan(y / x)
    elif x < 0 and y >= 0:
        return atan(y / x) + π
    elif x < 0 and y < 0:
        return atan(y / x) - π
    elif x == 0 and y > 0:
        return π/2
    elif x == 0 and y < 0:
        return -π/2
    else:  # x == 0 and y == 0
        return undefined (usually 0)
```

**Mathematical Properties**:
- **Full circle coverage**: Returns angles in (-π, π]
- **Quadrant-aware**: Correctly handles all four quadrants
- **Sign of arguments matters**: atan2(y, x) ≠ atan2(-y, -x)
- **Special values**:
  - atan2(0, 1) = 0 (positive x-axis)
  - atan2(1, 0) = π/2 (positive y-axis)
  - atan2(0, -1) = π (negative x-axis)
  - atan2(-1, 0) = -π/2 (negative y-axis)
  - atan2(1, 1) = π/4 (first quadrant, 45°)
  - atan2(1, -1) = 3π/4 (second quadrant, 135°)

**Quadrant Rules**:
```
Quadrant I   (x>0, y>0):  θ ∈ (0, π/2)
Quadrant II  (x<0, y>0):  θ ∈ (π/2, π)
Quadrant III (x<0, y<0):  θ ∈ (-π, -π/2)
Quadrant IV  (x>0, y<0):  θ ∈ (-π/2, 0)
```

**CPU Implementation** (native/BinaryOps.cpp):
```cpp
void atan2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "atan2_cpu", [&] {
    cpu_kernel_vec(
      iter,
      [](scalar_t y, scalar_t x) -> scalar_t {
        return std::atan2(y, x);
      },
      [](Vectorized<scalar_t> y, Vectorized<scalar_t> x) {
        return y.atan2(x);
      }
    );
  });
}
```

**MPS Implementation** (native/mps/operations/BinaryOps.mm):
```objc
Tensor atan2_mps(const Tensor& y, const Tensor& x) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* yTensor = getMPSGraphTensor(y);
  MPSGraphTensor* xTensor = getMPSGraphTensor(x);
  
  MPSGraphTensor* result = [mpsGraph atan2WithYTensor:yTensor
                                             xTensor:xTensor
                                                name:@"atan2"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
Let θ = atan2(y, x)

∂θ/∂y = x / (x² + y²)
∂θ/∂x = -y / (x² + y²)

Mathematical derivation:
From tan(θ) = y/x:
sec²(θ) · ∂θ/∂y = 1/x
∂θ/∂y = (1/x) / sec²(θ) = (1/x) · cos²(θ)

From geometry: cos(θ) = x / √(x²+y²)
Therefore: cos²(θ) = x² / (x²+y²)

∂θ/∂y = (1/x) · x²/(x²+y²) = x/(x²+y²)

Similarly for ∂θ/∂x:
sec²(θ) · ∂θ/∂x = -y/x²
∂θ/∂x = (-y/x²) · cos²(θ) = (-y/x²) · x²/(x²+y²) = -y/(x²+y²)

Chain rule:
If θ = atan2(f(t), g(t)), then:
dθ/dt = (∂θ/∂y)(df/dt) + (∂θ/∂x)(dg/dt)
      = [g/(f²+g²)](df/dt) + [-f/(f²+g²)](dg/dt)
```

**Backward Implementation**:
```cpp
std::tuple<Tensor, Tensor> atan2_backward(
    const Tensor& grad,
    const Tensor& y,  // self
    const Tensor& x   // other
) {
  auto r_sq = x.pow(2) + y.pow(2);
  
  // ∂atan2/∂y = x / (x² + y²)
  auto grad_y = grad * x / r_sq;
  
  // ∂atan2/∂x = -y / (x² + y²)
  auto grad_x = -grad * y / r_sq;
  
  return std::make_tuple(grad_y, grad_x);
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def atan2(y, x):
    """
    PyTorch: torch.atan2(y, x)
    MLX: mx.arctan2(y, x)
    
    Returns angle from positive x-axis to point (x, y)
    Range: (-π, π]
    """
    return mx.arctan2(y, x)

# Example 1: Full circle angles
y = mx.array([0.0, 1.0, 0.0, -1.0, 1.0, -1.0, -1.0, 1.0])
x = mx.array([1.0, 0.0, -1.0, 0.0, 1.0, 1.0, -1.0, -1.0])
angles = mx.arctan2(y, x)
# [0, π/2, π, -π/2, π/4, -π/4, -3π/4, 3π/4]
# All quadrants covered!

# Example 2: Convert Cartesian to polar coordinates
def cartesian_to_polar(x, y):
    """
    Convert (x, y) to (r, θ)
    r: radius
    θ: angle from positive x-axis in radians
    """
    r = mx.sqrt(x**2 + y**2)
    theta = mx.arctan2(y, x)
    return r, theta

points_x = mx.array([1.0, -1.0, -1.0, 1.0])
points_y = mx.array([1.0, 1.0, -1.0, -1.0])
r, theta = cartesian_to_polar(points_x, points_y)
# r = [√2, √2, √2, √2]
# theta = [π/4, 3π/4, -3π/4, -π/4]

# Example 3: Heading direction from velocity
def velocity_to_heading(vx, vy):
    """
    Compute heading angle from velocity components
    Returns angle in degrees from North (positive y-axis)
    """
    # atan2(vx, vy) because North is positive y
    heading_rad = mx.arctan2(vx, vy)
    heading_deg = mx.degrees(heading_rad)
    
    # Convert to compass bearing (0-360, clockwise from North)
    bearing = (90 - heading_deg) % 360
    return bearing

# Example 4: Rotation angle between two vectors
def angle_between_vectors_signed(v1, v2):
    """
    Compute signed angle from v1 to v2
    Returns angle in (-π, π]
    """
    # For 2D vectors
    cross = v1[0] * v2[1] - v1[1] * v2[0]  # Cross product z-component
    dot = v1[0] * v2[0] + v1[1] * v2[1]    # Dot product
    return mx.arctan2(cross, dot)

v1 = mx.array([1.0, 0.0])
v2 = mx.array([0.0, 1.0])
angle = angle_between_vectors_signed(v1, v2)  # π/2 (counterclockwise)
```

**Common Patterns**:
```python
# Pattern 1: Cartesian to polar conversion (most common)
def cart2pol(x, y):
    """Convert Cartesian to polar coordinates"""
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi

# Pattern 2: Optical flow angle
def flow_angle(flow_x, flow_y):
    """
    Compute flow direction from flow field
    flow_x, flow_y: [H, W] flow components
    Returns: [H, W] angles in radians
    """
    return torch.atan2(flow_y, flow_x)

# Pattern 3: Phase of complex number
def complex_phase(real, imag):
    """
    Compute phase angle of complex number
    z = real + i*imag
    phase = atan2(imag, real)
    """
    return torch.atan2(imag, real)

# Example in frequency domain
fft_result = torch.fft.fft(signal)
magnitude = torch.abs(fft_result)
phase = torch.atan2(fft_result.imag, fft_result.real)

# Pattern 4: Steering angle in robotics
class DifferentialDrive:
    def compute_steering_angle(self, target_x, target_y, robot_x, robot_y, robot_heading):
        """
        Compute steering angle to target
        """
        # Vector to target
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Absolute angle to target
        angle_to_target = torch.atan2(dy, dx)
        
        # Relative angle (steering command)
        steering_angle = angle_to_target - robot_heading
        
        # Normalize to (-π, π]
        steering_angle = torch.atan2(
            torch.sin(steering_angle),
            torch.cos(steering_angle)
        )
        
        return steering_angle

# Pattern 5: Gradient direction in images
def gradient_direction(image):
    """
    Compute gradient direction at each pixel
    Returns angles in (-π, π]
    """
    # Compute gradients (Sobel, etc.)
    grad_x = sobel_x(image)
    grad_y = sobel_y(image)
    
    # Direction
    direction = torch.atan2(grad_y, grad_x)
    return direction
```

**Edge Cases**:
- **Origin (0, 0)**: atan2(0, 0) typically returns 0 (implementation-defined)
- **Positive x-axis**: atan2(0, +x) = 0
- **Negative x-axis**: atan2(±0, -x) = ±π (sign matters!)
- **Positive y-axis**: atan2(+y, 0) = π/2
- **Negative y-axis**: atan2(-y, 0) = -π/2
- **Broadcasting**: Works element-wise with broadcasting
- **NaN propagation**: atan2(NaN, x) = NaN, atan2(y, NaN) = NaN

**Discontinuity**:
```python
# atan2 has a discontinuity along negative x-axis
# Jumps from π to -π

points_y = [0.1, 0.01, 0.001, -0.001, -0.01, -0.1]
points_x = [-1.0] * 6

angles = [atan2(y, x) for y, x in zip(points_y, points_x)]
# [3.04, 3.13, 3.14, -3.14, -3.13, -3.04]
#                   ↑ Jump from π to -π
```

**Performance Notes**:
- More expensive than atan (quadrant determination)
- Vectorized SIMD operations
- Branch-free implementations for performance
- Pointwise operation eligible for fusion

**MLX Porting Considerations**:
- MLX uses `arctan2` (not `atan2`)
- Identical semantics
- Same quadrant handling
- Same (y, x) argument order

---

## Summary - Week 4 Day 2

**Total Operators Documented**: 7 trigonometric functions (3 basic + 4 inverse)

**Day 1 - Basic Trig** (3 ops): sin, cos, tan ✅
**Day 2 - Inverse Trig** (4 ops): asin, acos, atan, atan2 ✅

**Inverse Function Properties**:

| Function | Domain | Range | Derivative | Notes |
|----------|--------|-------|------------|-------|
| asin(x) | [-1, 1] | [-π/2, π/2] | 1/√(1-x²) | Gradient → ∞ at ±1 |
| acos(x) | [-1, 1] | [0, π] | -1/√(1-x²) | Always positive output |
| atan(x) | ℝ | (-π/2, π/2) | 1/(1+x²) | Bounded gradient |
| atan2(y,x) | ℝ² | (-π, π] | See above | Full circle, quadrant-aware |

**Key Distinctions**:
- **asin/acos**: Restricted domain [-1, 1], gradient explodes at boundaries
- **atan**: Unrestricted domain, well-behaved gradients
- **atan2**: Two-argument, full circle coverage, preferred for angle computation

**Critical Usage Notes**:
- **Always clamp inputs** to asin/acos: `torch.clamp(x, -1+ε, 1-ε)`
- **Use atan2, not atan** for angle from components (avoids quadrant ambiguity)
- **Gradient instability** in asin/acos near ±1 can cause training issues

**Common Applications**:
- **asin/acos**: Angle recovery, spherical coordinates, vector angles
- **atan**: Slope to angle (limited range), smooth activation
- **atan2**: Polar coordinates, heading/bearing, phase angles, optical flow

**PyTorch → MLX Mapping**:
- `torch.asin` → `mx.arcsin`
- `torch.acos` → `mx.arccos`
- `torch.atan` → `mx.arctan`
- `torch.atan2` → `mx.arctan2`

**Progress**: 7 / 15 trigonometric operators documented (47%)
**Week 4 Day 1**: sin, cos, tan ✅
**Week 4 Day 2**: asin, acos, atan, atan2 ✅
**Week 4 Day 3**: sinh, cosh, tanh (pending)
**Week 4 Day 4**: asinh, acosh, atanh (pending)
**Week 4 Day 5**: deg2rad, rad2deg (pending)


---

## Week 4 Day 3 Operators - Hyperbolic Trigonometric Functions

### sinh

**Purpose**: Compute hyperbolic sine of input (element-wise)

**Signature**: `sinh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:5534-5541):
```yaml
- func: sinh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: sinh.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: sinh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: sinh_sparse_csr
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = sinh(self[i])
# Defined as: sinh(x) = (e^x - e^(-x)) / 2
# Domain: (-∞, ∞)
# Range: (-∞, ∞)

result[i] = (exp(self[i]) - exp(-self[i])) / 2
```

**Mathematical Properties**:
- **Definition**: sinh(x) = (e^x - e^(-x)) / 2
- **Symmetry**: Odd function, sinh(-x) = -sinh(x)
- **Unbounded**: No restrictions on input or output
- **Special values**:
  - sinh(0) = 0
  - sinh(1) ≈ 1.175
  - sinh(-1) ≈ -1.175
- **Growth**: Exponential growth for large |x|

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void sinh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "sinh_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::sinh(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.sinh();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor sinh_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph sinhWithTensor:selfTensor
                                               name:@"sinh"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂sinh(x)/∂x = cosh(x)

Mathematical derivation:
Let y = sinh(x) = (e^x - e^(-x))/2

dy/dx = d/dx[(e^x - e^(-x))/2]
      = (e^x + e^(-x))/2
      = cosh(x)

Chain rule:
If z = sinh(f(x)), then:
∂z/∂x = cosh(f(x)) · ∂f/∂x
```

**Backward Implementation**:
```cpp
Tensor sinh_backward(const Tensor& grad, const Tensor& input) {
  // ∂sinh/∂x = cosh(x)
  return grad * input.cosh();
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def sinh(x):
    """
    PyTorch: torch.sinh(x)
    MLX: mx.sinh(x)
    """
    return mx.sinh(x)

# Example 1: Basic hyperbolic sine
values = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = mx.sinh(values)
# [-3.627, -1.175, 0.0, 1.175, 3.627]

# Example 2: Manual computation
x = mx.array([1.0])
sinh_manual = (mx.exp(x) - mx.exp(-x)) / 2
sinh_builtin = mx.sinh(x)
# Both give same result

# Example 3: Scaling activation (less common than tanh)
class SinhActivation(nn.Module):
    """Hyperbolic sine activation (rarely used, unbounded)"""
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        # Scale down to avoid overflow
        return torch.sinh(x * self.scale)

# Example 4: Physical applications (catenary curve)
def catenary(x, a=1.0):
    """
    Catenary curve (shape of hanging chain)
    y = a * cosh(x/a)
    Related to sinh for certain transformations
    """
    return a * mx.cosh(x / a)
```

**Common Patterns**:
```python
# Pattern 1: Swish/SiLU activation implementation
# Note: Modern implementations use different formulas,
# but sinh-based versions exist
def sinh_swish(x):
    """Swish-like activation using sinh"""
    # Standard Swish: x * sigmoid(x)
    # Sinh variant (less common):
    return x * torch.tanh(x)  # Uses tanh, related to sinh/cosh

# Pattern 2: Numerical range transformation
def sinh_stretch(x, scale=1.0):
    """
    Stretch values using sinh (preserves sign, increases range)
    """
    return torch.sinh(x / scale) * scale

# Pattern 3: Symmetric unbounded activation
class SinhActivation(nn.Module):
    """
    Unbounded symmetric activation
    - Preserves sign
    - Smooth at origin
    - Exponential growth
    """
    def forward(self, x):
        return torch.sinh(x)

# Pattern 4: Physics/engineering calculations
def relativistic_velocity_addition(v1, v2, c=1.0):
    """
    Relativistic velocity addition using hyperbolic functions
    v1, v2: velocities
    c: speed of light
    """
    # Rapidity: θ = atanh(v/c)
    theta1 = torch.atanh(v1 / c)
    theta2 = torch.atanh(v2 / c)
    
    # Add rapidities
    theta_total = theta1 + theta2
    
    # Convert back: v = c * tanh(θ)
    v_total = c * torch.tanh(theta_total)
    return v_total
```

**Edge Cases**:
- **Large inputs**: sinh(x) ≈ e^x/2 for large positive x (overflow risk)
- **Overflow**: sinh(x) overflows to inf for x > ~710 (float64)
- **Symmetry**: sinh(x) = -sinh(-x)
- **NaN**: sinh(NaN) = NaN
- **Inf**: sinh(±∞) = ±∞

**Numerical Stability**:
- **For large |x|**: Can overflow
- **Implementation**: Often uses `sinh(x) = sign(x) * (exp(|x|) - exp(-|x|))/2`
- **Recommendation**: Scale inputs if possible to avoid overflow

**Performance Notes**:
- More expensive than polynomial activations
- Requires exp computation (expensive)
- Vectorized SIMD operations
- Sparse tensor support

**MLX Porting Considerations**:
- Direct equivalent: `mx.sinh()`
- Identical semantics
- Same overflow characteristics

---

### cosh

**Purpose**: Compute hyperbolic cosine of input (element-wise)

**Signature**: `cosh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:1853-1857):
```yaml
- func: cosh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  structured_delegate: cosh.out
  tags: [core, pointwise]
```

**Algorithm**:
```python
# Element-wise: result[i] = cosh(self[i])
# Defined as: cosh(x) = (e^x + e^(-x)) / 2
# Domain: (-∞, ∞)
# Range: [1, ∞)  # Always >= 1

result[i] = (exp(self[i]) + exp(-self[i])) / 2
```

**Mathematical Properties**:
- **Definition**: cosh(x) = (e^x + e^(-x)) / 2
- **Symmetry**: Even function, cosh(-x) = cosh(x)
- **Minimum value**: cosh(0) = 1 (global minimum)
- **Special values**:
  - cosh(0) = 1
  - cosh(1) ≈ 1.543
  - cosh(-1) ≈ 1.543 (same as cosh(1))
- **Growth**: Exponential growth for large |x|
- **Identity**: cosh²(x) - sinh²(x) = 1

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void cosh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "cosh_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::cosh(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.cosh();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor cosh_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph coshWithTensor:selfTensor
                                               name:@"cosh"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂cosh(x)/∂x = sinh(x)

Mathematical derivation:
Let y = cosh(x) = (e^x + e^(-x))/2

dy/dx = d/dx[(e^x + e^(-x))/2]
      = (e^x - e^(-x))/2
      = sinh(x)

Chain rule:
If z = cosh(f(x)), then:
∂z/∂x = sinh(f(x)) · ∂f/∂x
```

**Backward Implementation**:
```cpp
Tensor cosh_backward(const Tensor& grad, const Tensor& input) {
  // ∂cosh/∂x = sinh(x)
  return grad * input.sinh();
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def cosh(x):
    """
    PyTorch: torch.cosh(x)
    MLX: mx.cosh(x)
    """
    return mx.cosh(x)

# Example 1: Basic hyperbolic cosine
values = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
result = mx.cosh(values)
# [3.762, 1.543, 1.0, 1.543, 3.762]
# Note: symmetric around 0

# Example 2: Catenary curve (hanging chain/cable)
def catenary_curve(x, a=1.0, x0=0.0):
    """
    Shape of a hanging flexible chain/cable
    a: parameter determining sag
    x0: horizontal offset
    """
    return a * mx.cosh((x - x0) / a)

x = mx.linspace(-5, 5, 100)
y = catenary_curve(x, a=2.0)
# Produces characteristic U-shape

# Example 3: Relationship to circular functions
def verify_hyperbolic_identity(x):
    """
    Verify: cosh²(x) - sinh²(x) = 1
    (Analogous to cos²(x) + sin²(x) = 1)
    """
    lhs = mx.cosh(x)**2 - mx.sinh(x)**2
    return mx.allclose(lhs, mx.ones_like(x))

# Example 4: Distance in hyperbolic space
def hyperbolic_distance(x1, y1, x2, y2):
    """
    Distance in Poincaré half-plane model
    Uses cosh in distance formula
    """
    dx = x2 - x1
    dy = y2 - y1
    
    # Hyperbolic distance formula
    d = mx.arccosh(1 + (dx**2 + (y2 - y1)**2) / (2 * y1 * y2))
    return d
```

**Common Patterns**:
```python
# Pattern 1: Catenary curve (physics/engineering)
def catenary(x, T, w, x0=0):
    """
    Exact catenary curve
    T: horizontal tension
    w: weight per unit length
    """
    a = T / w
    return a * torch.cosh((x - x0) / a)

# Pattern 2: Gaussian-like bump function
def cosh_bump(x, width=1.0):
    """
    Smooth bump function using cosh
    Smoother than Gaussian at tails
    """
    return 1.0 / torch.cosh(x / width)

# Pattern 3: Hyperbolic angle parameterization
def hyperbolic_rotation(x, y, angle):
    """
    Hyperbolic rotation (Lorentz boost in physics)
    Similar to circular rotation but uses cosh/sinh
    """
    x_new = x * torch.cosh(angle) + y * torch.sinh(angle)
    y_new = x * torch.sinh(angle) + y * torch.cosh(angle)
    return x_new, y_new

# Pattern 4: Soft maximum approximation
def soft_maximum(x, y, alpha=1.0):
    """
    Smooth approximation to max(x, y) using hyperbolic functions
    As alpha → ∞, approaches true max
    """
    return (x + y + alpha * torch.log(
        torch.cosh((x - y) / alpha)
    )) / 2
```

**Edge Cases**:
- **Minimum**: cosh(0) = 1 (always >= 1)
- **Large |x|**: cosh(x) ≈ e^|x|/2 for large |x| (overflow risk)
- **Overflow**: cosh(x) overflows for |x| > ~710 (float64)
- **Symmetry**: cosh(x) = cosh(-x)
- **NaN**: cosh(NaN) = NaN
- **Inf**: cosh(±∞) = +∞

**Numerical Stability**:
- **For large |x|**: Can overflow
- **Implementation**: `cosh(x) = (exp(|x|) + exp(-|x|))/2`
- **Recommendation**: Be aware of overflow for |x| > 700

**Performance Notes**:
- Requires exp computation (expensive)
- Vectorized SIMD operations
- Similar cost to sinh

**MLX Porting Considerations**:
- Direct equivalent: `mx.cosh()`
- Identical semantics
- Same overflow behavior

---

### tanh

**Purpose**: Compute hyperbolic tangent of input (element-wise)

**Signature**: `tanh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:6211-6220):
```yaml
- func: tanh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: tanh.out
  variants: function, method
  dispatch:
    QuantizedCPU: tanh_quantized_cpu
    MkldnnCPU: mkldnn_tanh
    SparseCPU, SparseCUDA, SparseMPS: tanh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: tanh_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_tanh
```

**Algorithm**:
```python
# Element-wise: result[i] = tanh(self[i])
# Defined as: tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# Domain: (-∞, ∞)
# Range: (-1, 1)  # Bounded!

result[i] = (exp(2*self[i]) - 1) / (exp(2*self[i]) + 1)
# Or equivalently:
result[i] = 2 / (1 + exp(-2*self[i])) - 1
```

**Mathematical Properties**:
- **Definition**: tanh(x) = sinh(x)/cosh(x) = (e^x - e^(-x))/(e^x + e^(-x))
- **Symmetry**: Odd function, tanh(-x) = -tanh(x)
- **Bounded output**: Range is (-1, 1), never reaches ±1
- **S-shaped curve**: Sigmoid-like but symmetric around origin
- **Special values**:
  - tanh(0) = 0
  - tanh(1) ≈ 0.762
  - tanh(-1) ≈ -0.762
  - tanh(∞) = 1
  - tanh(-∞) = -1
- **Relationship to sigmoid**: tanh(x) = 2·sigmoid(2x) - 1

**CPU Implementation** (native/UnaryOps.cpp):
```cpp
void tanh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
    kHalf, kBFloat16, iter.dtype(), "tanh_cpu", [&] {
      cpu_kernel_vec(
        iter,
        [](scalar_t x) -> scalar_t {
          return std::tanh(x);
        },
        [](Vectorized<scalar_t> x) {
          return x.tanh();
        }
      );
    }
  );
}
```

**MPS Implementation** (native/mps/operations/UnaryOps.mm):
```objc
Tensor tanh_mps(const Tensor& self) {
  MPSGraph* mpsGraph = make_mps_graph();
  MPSGraphTensor* selfTensor = getMPSGraphTensor(self);
  
  MPSGraphTensor* result = [mpsGraph tanhWithTensor:selfTensor
                                               name:@"tanh"];
  
  return createTensorFromMPSGraph(result);
}
```

**Backward Pass**:
```
∂tanh(x)/∂x = 1 - tanh²(x) = sech²(x) = 1/cosh²(x)

Mathematical derivation:
Let y = tanh(x) = sinh(x)/cosh(x)

Using quotient rule:
dy/dx = [cosh(x)·cosh(x) - sinh(x)·sinh(x)] / cosh²(x)
      = [cosh²(x) - sinh²(x)] / cosh²(x)
      = 1 / cosh²(x)    [using cosh²(x) - sinh²(x) = 1]
      = sech²(x)

Alternative form (more efficient):
Since tanh(x) = sinh(x)/cosh(x), we have:
sinh²(x) + 1 = cosh²(x)
sinh²(x)/cosh²(x) + 1/cosh²(x) = 1
tanh²(x) + sech²(x) = 1
sech²(x) = 1 - tanh²(x)

Therefore:
dy/dx = 1 - tanh²(x)

Chain rule:
If z = tanh(f(x)), then:
∂z/∂x = (1 - tanh²(f(x))) · ∂f/∂x
```

**Backward Implementation**:
```cpp
Tensor tanh_backward(const Tensor& grad, const Tensor& output) {
  // ∂tanh/∂x = 1 - tanh²(x)
  // Using output (already computed tanh) is more efficient
  return grad * (1 - output.pow(2));
  
  // Alternative (less efficient):
  // auto tanh_x = input.tanh();
  // return grad * (1 - tanh_x.pow(2));
}
```

**MLX Equivalent**:
```python
import mlx.core as mx

def tanh(x):
    """
    PyTorch: torch.tanh(x)
    MLX: mx.tanh(x)
    """
    return mx.tanh(x)

# Example 1: Basic hyperbolic tangent
values = mx.array([-5.0, -1.0, 0.0, 1.0, 5.0])
result = mx.tanh(values)
# [-0.9999, -0.762, 0.0, 0.762, 0.9999]
# Saturates to ±1 for large |x|

# Example 2: Activation function (most common use)
class TanhActivation(nn.Module):
    """
    Tanh activation layer
    Popular before ReLU, still used in RNNs/LSTMs
    """
    def forward(self, x):
        return torch.tanh(x)

# Example 3: LSTM gate computation
class LSTMCell:
    def forward(self, x, h_prev, c_prev):
        # Gates typically use tanh for cell state
        gates = self.linear(torch.cat([x, h_prev], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Cell gate (tanh!)
        o = torch.sigmoid(o)  # Output gate
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)  # Hidden state (tanh!)
        
        return h, c

# Example 4: Relationship to sigmoid
def tanh_from_sigmoid(x):
    """tanh(x) = 2·sigmoid(2x) - 1"""
    return 2 * mx.sigmoid(2 * x) - 1

def sigmoid_from_tanh(x):
    """sigmoid(x) = (1 + tanh(x/2)) / 2"""
    return (1 + mx.tanh(x / 2)) / 2
```

**Common Patterns**:
```python
# Pattern 1: Classic activation function (pre-ReLU era)
class MLPClassic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # Tanh activation
        x = self.fc2(x)
        return x

# Pattern 2: LSTM/GRU (still widely used)
class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Gates: input, forget, cell, output
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)
    
    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=-1)
        gates = self.W(combined)
        
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)  # Cell candidate
        o = torch.sigmoid(o)
        
        c = f * c_prev + i * g
        h = o * torch.tanh(c)  # Output
        return h, (h, c)

# Pattern 3: Attention (value normalization)
def scaled_dot_product_attention_tanh(Q, K, V, scale=None):
    """
    Attention variant using tanh (less common than softmax)
    """
    if scale is None:
        scale = Q.size(-1) ** 0.5
    
    scores = (Q @ K.transpose(-2, -1)) / scale
    # Some variants use tanh instead of softmax
    attention = torch.tanh(scores)  # Range (-1, 1)
    attention = (attention + 1) / 2  # Shift to (0, 1)
    
    return attention @ V

# Pattern 4: Gradient clipping via tanh
def tanh_gradient_clip(x, clip_value=10.0):
    """
    Soft gradient clipping using tanh
    More gradual than hard clipping
    """
    return clip_value * torch.tanh(x / clip_value)

# Pattern 5: Output layer for bounded targets
class BoundedRegression(nn.Module):
    def __init__(self, input_size, min_val=-1.0, max_val=1.0):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        self.min_val = min_val
        self.max_val = max_val
    
    def forward(self, x):
        # Map to (min_val, max_val)
        x = self.fc(x)
        x = torch.tanh(x)  # (-1, 1)
        # Scale and shift
        x = (x + 1) / 2  # (0, 1)
        x = x * (self.max_val - self.min_val) + self.min_val
        return x
```

**Edge Cases**:
- **Large positive**: tanh(x) → 1 as x → ∞ (saturates)
- **Large negative**: tanh(x) → -1 as x → -∞ (saturates)
- **Zero**: tanh(0) = 0
- **Gradient vanishing**: Gradient ≈ 0 for |x| > 3 (saturation problem)
- **NaN**: tanh(NaN) = NaN
- **Inf**: tanh(±∞) = ±1

**Comparison to ReLU and Sigmoid**:
```python
# tanh: Range (-1, 1), symmetric, saturates at both ends
torch.tanh(x)

# ReLU: Range [0, ∞), not symmetric, no saturation for x > 0
torch.relu(x)

# Sigmoid: Range (0, 1), not symmetric, saturates at both ends
torch.sigmoid(x)

# Gradient comparison at x = 0:
# tanh'(0) = 1
# relu'(0) = 1
# sigmoid'(0) = 0.25

# Gradient at x = 3:
# tanh'(3) ≈ 0.01 (vanishing!)
# relu'(3) = 1 (constant)
# sigmoid'(3) ≈ 0.045 (vanishing!)
```

**Vanishing Gradient Problem**:
- **Issue**: For |x| > 2, gradient < 0.1
- **Impact**: Deep networks with tanh suffer from vanishing gradients
- **Solution**: ReLU and variants (modern preference)
- **Still used**: LSTMs/GRUs (gate functions benefit from bounded outputs)

**Performance Notes**:
- More expensive than ReLU (exp computation)
- Often optimized with rational approximations
- Vectorized SIMD operations
- Quantized implementation available
- Sparse tensor support

**MLX Porting Considerations**:
- Direct equivalent: `mx.tanh()`
- Identical semantics
- Same saturation behavior
- Critical for LSTM/GRU implementations

---

## Summary - Week 4 Day 3

**Total Operators Documented**: 10 trigonometric functions (7 regular + 3 hyperbolic)

**Day 1 - Basic Trig** (3 ops): sin, cos, tan ✅
**Day 2 - Inverse Trig** (4 ops): asin, acos, atan, atan2 ✅
**Day 3 - Hyperbolic Trig** (3 ops): sinh, cosh, tanh ✅

**Hyperbolic Function Properties**:

| Function | Definition | Domain | Range | Derivative |
|----------|------------|--------|-------|------------|
| sinh(x) | (e^x - e^(-x))/2 | ℝ | ℝ | cosh(x) |
| cosh(x) | (e^x + e^(-x))/2 | ℝ | [1, ∞) | sinh(x) |
| tanh(x) | sinh/cosh | ℝ | (-1, 1) | 1 - tanh²(x) |

**Key Relationships**:
- cosh²(x) - sinh²(x) = 1 (hyperbolic identity)
- tanh(x) = sinh(x)/cosh(x)
- tanh(x) = 2·sigmoid(2x) - 1
- sinh(-x) = -sinh(x) (odd)
- cosh(-x) = cosh(x) (even)
- tanh(-x) = -tanh(x) (odd)

**Circular vs Hyperbolic**:
```
Circular:              Hyperbolic:
cos²(x) + sin²(x) = 1  cosh²(x) - sinh²(x) = 1
Range of sin/cos: [-1,1]  Range of tanh: (-1,1), cosh: [1,∞)
Periodic           Not periodic
```

**Common Applications**:
- **sinh/cosh**: Physics (catenary curves, relativity), hyperbolic geometry
- **tanh**: Activation functions (RNNs, LSTMs), bounded outputs, sigmoid alternative

**Activation Function Comparison**:
```
tanh:
- Pros: Bounded (-1,1), zero-centered, smooth
- Cons: Vanishing gradient for |x| > 2, expensive (exp)
- Use: LSTMs, GRUs, classical NNs

ReLU:
- Pros: No vanishing gradient (x>0), cheap, sparse activation
- Cons: Not bounded, not zero-centered, dying ReLU
- Use: Modern CNNs, most hidden layers

Sigmoid:
- Pros: Bounded (0,1), interpretable as probability
- Cons: Not zero-centered, vanishing gradient, expensive
- Use: Binary classification output, gates
```

**PyTorch → MLX Mapping**:
- `torch.sinh` → `mx.sinh`
- `torch.cosh` → `mx.cosh`
- `torch.tanh` → `mx.tanh`

**Progress**: 10 / 15 trigonometric operators documented (67%)
**Week 4 Day 1**: sin, cos, tan ✅
**Week 4 Day 2**: asin, acos, atan, atan2 ✅
**Week 4 Day 3**: sinh, cosh, tanh ✅
**Week 4 Day 4**: asinh, acosh, atanh (pending)
**Week 4 Day 5**: deg2rad, rad2deg (pending)


---

## Week 4 Day 4 Operators - Inverse Hyperbolic Functions

### asinh

**Purpose**: Compute inverse hyperbolic sine (element-wise)

**Signature**: `asinh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:888-894):
```yaml
- func: asinh(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: asinh.out
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: asinh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: asinh_sparse_csr
  tags: [core, pointwise]
```

**Algorithm**: `result[i] = log(self[i] + sqrt(self[i]² + 1))`
**Domain**: ℝ (all real numbers)
**Range**: ℝ
**Derivative**: `∂asinh(x)/∂x = 1 / sqrt(x² + 1)`

**MLX Equivalent**: `mx.arcsing(x)`

**Common Use**: Stabilizing transformations, handling unbounded inputs

---

### acosh

**Purpose**: Compute inverse hyperbolic cosine (element-wise)

**Signature**: `acosh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:861-864):
```yaml
- func: acosh(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: acosh.out
  tags: [core, pointwise]
```

**Algorithm**: `result[i] = log(self[i] + sqrt(self[i]² - 1))`
**Domain**: [1, ∞) (requires x >= 1)
**Range**: [0, ∞)
**Derivative**: `∂acosh(x)/∂x = 1 / sqrt(x² - 1)`

**MLX Equivalent**: `mx.arccosh(x)`

**Common Use**: Hyperbolic geometry, physics calculations

---

### atanh

**Purpose**: Compute inverse hyperbolic tangent (element-wise)

**Signature**: `atanh(Tensor self) -> Tensor`

**YAML Definition** (native_functions.yaml:923-929):
```yaml
- func: atanh(Tensor self) -> Tensor
  structured_delegate: atanh.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA, SparseMPS: atanh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: atanh_sparse_csr
  tags: [core, pointwise]
```

**Algorithm**: `result[i] = 0.5 * log((1 + self[i]) / (1 - self[i]))`
**Domain**: (-1, 1) (requires |x| < 1)
**Range**: ℝ
**Derivative**: `∂atanh(x)/∂x = 1 / (1 - x²)`

**MLX Equivalent**: `mx.arctanh(x)`

**Common Use**: Logit transformations, probability unbounding

**Edge Cases**:
- **Domain violations**: asinh (none), acosh (x < 1), atanh (|x| >= 1) return NaN
- **Gradient explosions**: acosh/atanh have unbounded gradients at domain boundaries

---

## Week 4 Day 5 Operators - Angle Conversions

### deg2rad

**Purpose**: Convert degrees to radians (element-wise)

**Signature**: `deg2rad(Tensor self) -> Tensor`

**Algorithm**: `result[i] = self[i] * π / 180`
**Domain**: ℝ
**Range**: ℝ
**Derivative**: `∂deg2rad(x)/∂x = π/180 ≈ 0.01745`

**MLX Equivalent**: `mx.radians(x)` or `x * mx.pi / 180`

**Example**:
```python
angles_deg = mx.array([0, 45, 90, 180, 360])
angles_rad = mx.radians(angles_deg)
# [0, π/4, π/2, π, 2π]
```

---

### rad2deg

**Purpose**: Convert radians to degrees (element-wise)

**Signature**: `rad2deg(Tensor self) -> Tensor`

**Algorithm**: `result[i] = self[i] * 180 / π`
**Domain**: ℝ
**Range**: ℝ
**Derivative**: `∂rad2deg(x)/∂x = 180/π ≈ 57.296`

**MLX Equivalent**: `mx.degrees(x)` or `x * 180 / mx.pi`

**Example**:
```python
angles_rad = mx.array([0, mx.pi/4, mx.pi/2, mx.pi])
angles_deg = mx.degrees(angles_rad)
# [0, 45, 90, 180]
```

**Common Pattern**:
```python
# Full conversion workflow
theta_deg = 45.0
theta_rad = torch.deg2rad(torch.tensor(theta_deg))
sin_theta = torch.sin(theta_rad)
result_rad = torch.asin(sin_theta)
result_deg = torch.rad2deg(result_rad)  # Back to 45.0
```

---

## Week 4 Complete Summary

**Total Operators Documented**: 15 trigonometric operators

**Day 1 - Basic Trig** (3 ops): sin, cos, tan ✅
**Day 2 - Inverse Trig** (4 ops): asin, acos, atan, atan2 ✅  
**Day 3 - Hyperbolic** (3 ops): sinh, cosh, tanh ✅
**Day 4 - Inverse Hyperbolic** (3 ops): asinh, acosh, atanh ✅
**Day 5 - Conversions** (2 ops): deg2rad, rad2deg ✅

**Function Family Overview**:

| Category | Functions | Domain Restrictions | Bounded Output |
|----------|-----------|-------------------|----------------|
| Basic Trig | sin, cos, tan | tan: x ≠ π/2+nπ | sin,cos: [-1,1] |
| Inverse Trig | asin, acos, atan, atan2 | asin,acos: [-1,1] | Yes |
| Hyperbolic | sinh, cosh, tanh | None | tanh: (-1,1) |
| Inv. Hyperbolic | asinh, acosh, atanh | acosh: [1,∞), atanh: (-1,1) | asinh: No |
| Conversions | deg2rad, rad2deg | None | No |

**Critical Safety Rules**:
1. **Always clamp** before asin/acos/atanh to avoid NaN
2. **Use atan2**, not atan, for angle from x/y components
3. **Beware vanishing gradients** in tanh for |x| > 2
4. **Avoid overflow** in sinh/cosh for |x| > 700

**PyTorch → MLX Naming**:
- PyTorch: `asin`, `acos`, `atan`, `atan2`
- MLX: `arcsin`, `arccos`, `arctan`, `arctan2` (NumPy convention)
- PyTorch: `asinh`, `acosh`, `atanh`
- MLX: `arcsinh`, `arccosh`, `arctanh`
- PyTorch: `deg2rad`, `rad2deg`
- MLX: `radians`, `degrees`

**Most Important for Deep Learning**:
1. **tanh**: LSTM/GRU gates, classical activations
2. **sin/cos**: Positional encodings (Transformers)
3. **atan2**: Angle computations, phase extraction
4. **deg2rad/rad2deg**: Human-readable angle I/O

**Progress**: 15 / 15 trigonometric operators documented (100%) ✅

**Week 4 Status**: Complete ✅

