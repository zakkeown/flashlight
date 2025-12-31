# Autograd Overview

## Purpose

PyTorch's autograd system provides **reverse-mode automatic differentiation** (also called backpropagation) by recording operations on tensors during the forward pass and constructing a computational graph. During the backward pass, this graph is traversed in reverse topological order to compute gradients via the chain rule.

Autograd is PyTorch's differentiating feature (pun intended) and is fundamental to:
- Training neural networks via gradient descent
- Computing gradients for optimization problems
- Custom gradient definitions via `torch.autograd.Function`
- Forward-mode AD for computing Jacobian-vector products

The system is **tape-based**: operations are recorded on-the-fly during forward execution, creating a dynamic computation graph that can change between forward passes.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyTorch Autograd System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Tensor     │         │   Tensor     │                     │
│  │ (Leaf Node)  │         │ (Interior)   │                     │
│  │              │         │              │                     │
│  │ requires_grad│         │  grad_fn ────┼──┐                  │
│  │   = True     │         │              │  │                  │
│  │              │         │   grad_ ─────┼──┼──> Tensor       │
│  │ grad_accumulator      │              │  │                  │
│  │     │        │         └──────────────┘  │                  │
│  └─────┼────────┘                           │                  │
│        │                                    │                  │
│        │                                    │                  │
│        ▼                                    ▼                  │
│  ┌────────────────┐              ┌────────────────┐            │
│  │AccumulateGrad  │              │   Node         │            │
│  │   (Sink Node)  │              │  (grad_fn)     │            │
│  │                │              │                │            │
│  │  operator()    │              │  apply()       │            │
│  │    └─> accumulates            │    └─> backward logic      │
│  │        into grad_              │                │            │
│  └────────────────┘              └────┬───────────┘            │
│                                       │                        │
│                                       │ next_edges[]           │
│                                       ▼                        │
│                              ┌─────────────────┐               │
│                              │     Edge        │               │
│                              │ (function, nr)  │               │
│                              └─────────────────┘               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Backward Engine                        │  │
│  │                                                          │  │
│  │   GraphTask ──> ReadyQueue ──> NodeTask execution       │  │
│  │      │             (priority queue)         │            │  │
│  │      │                                      │            │  │
│  │      └─> Topological sort & scheduling     │            │  │
│  │                                             ▼            │  │
│  │                               Call Node::operator()     │  │
│  │                               Accumulate gradients      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### AutogradMeta

**File**: [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h#L227-L358)

Every Tensor that participates in autograd has an `AutogradMeta` structure (or a nullptr for tensors that don't need autograd). This structure stores all autograd-specific metadata.

```cpp
struct AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;                          // Optional debug name
  Variable grad_;                              // The gradient accumulator
  std::shared_ptr<Node> grad_fn_;             // Gradient function (for interior nodes)
  std::weak_ptr<Node> grad_accumulator_;      // Gradient accumulator (for leaf nodes)
  std::shared_ptr<ForwardGrad> fw_grad_;      // Forward-mode AD gradients

  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;
  std::unique_ptr<PostAccumulateGradHook> post_acc_grad_hooks_;

  bool requires_grad_{false};   // Only meaningful for leaf variables
  bool retains_grad_{false};    // Only meaningful for non-leaf variables
  bool is_view_{false};

  uint32_t output_nr_;          // Which output of grad_fn this tensor is

  std::optional<at::ScalarType> grad_dtype_;
  bool allow_grad_dtype_mismatch_{false};

  mutable std::mutex mutex_;    // Thread-safety for concurrent access

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_;
  }

  Variable& mutable_grad() override { return grad_; }
  const Variable& grad() const override { return grad_; }
};
```

**Key Fields**:

- **`grad_`**: Stores the accumulated gradient for this tensor. For leaf tensors, this is where gradients accumulate during backward. For non-leaf tensors, this is only populated if `retain_grad()` is called.

- **`grad_fn_`**: Shared pointer to the `Node` that created this tensor. This is the function whose backward will be called during backprop. Only set for non-leaf tensors.

- **`grad_accumulator_`**: Weak pointer to the `AccumulateGrad` node for leaf tensors. This is what actually writes gradients into `grad_`.

- **`requires_grad_`**: Only true for leaf tensors that explicitly request gradient tracking. Interior tensors determine `requires_grad()` based on whether they have a `grad_fn_`.

- **`output_nr_`**: If a function produces multiple outputs, this identifies which output this tensor is (0-indexed). Used to route gradients to the correct input during backward.

### Variable (Tensor)

**File**: [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h#L25-L33)

```cpp
using Variable = at::Tensor;
```

Historically, PyTorch distinguished between `Variable` (with autograd) and `Tensor` (without). Now they are **exactly the same type**. The name `Variable` is kept only for backward compatibility.

**Conceptual Categories**:

1. **Leaf Variables**: Created directly by the user or don't have a gradient function
   - `requires_grad` explicitly set to `True`
   - Have a `grad_accumulator` but no `grad_fn`
   - Examples: model parameters, input data with `requires_grad=True`

2. **Interior Variables**: Result of operations on tensors that require gradients
   - Have a `grad_fn` pointing to the operation that created them
   - Don't accumulate gradients unless `retain_grad()` is called
   - Examples: intermediate activations in a neural network

3. **Non-differentiable Tensors**: Don't participate in autograd
   - `autograd_meta` is `nullptr` (optimization)
   - No gradient tracking overhead
   - Examples: inputs without `requires_grad`, intermediate results when `torch.no_grad()` is active

**Leaf vs Interior Decision**:

```python
# Leaf examples
x = torch.tensor([1.0, 2.0], requires_grad=True)  # Leaf
assert x.is_leaf == True
assert x.grad_fn is None

# Interior examples
y = x * 2                                          # Interior
assert y.is_leaf == False
assert y.grad_fn is not None  # MulBackward0
```

### Edge

**File**: [torch/csrc/autograd/edge.h](../reference/pytorch/torch/csrc/autograd/edge.h#L14-L39)

An `Edge` represents a connection in the computational graph from a `Node` to one of its inputs.

```cpp
struct Edge {
  std::shared_ptr<Node> function;  // The target Node
  uint32_t input_nr;                // Which input of the Node

  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  bool is_valid() const noexcept {
    return function != nullptr;
  }
};
```

**Purpose**:
- Connect tensors to gradient functions
- Connect gradient functions to each other
- Route gradients to the correct input during backward

**Example**:
```python
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = x * y

# z.grad_fn points to MulBackward0
# MulBackward0 has two next_edges:
#   Edge(AccumulateGrad for x, input_nr=0)
#   Edge(AccumulateGrad for y, input_nr=0)
```

### Node (Gradient Function)

**File**: [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h#L113-L705)

`Node` is the abstract base class for all gradient functions. Every differentiable operation creates a `Node` subclass instance.

```cpp
struct Node : std::enable_shared_from_this<Node> {
  // Graph connectivity
  edge_list next_edges_;                      // Outgoing edges to next nodes
  at::SmallVector<InputMetadata, 2> input_metadata_;  // Shape/dtype info for inputs

  // Execution order
  uint64_t sequence_nr_;                      // When was this node created?
  uint64_t topological_nr_;                   // Longest path to leaf
  mutable bool has_parent_;                   // Has this been added to a graph?
  uint64_t thread_id_;                        // Which thread created this?

  // Hooks
  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;
  std::vector<std::unique_ptr<FunctionPreHook>> tensor_pre_hooks_;
  std::unordered_map<size_t, std::unique_ptr<FunctionPreHook>> retains_grad_hooks_;
  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;

  // Thread safety
  std::mutex mutex_;

  // Python integration
  PyObject* pyobj_;                           // Weak reference to Python object
  std::unique_ptr<AnomalyMetadata> anomaly_metadata_;

  // Core API
  variable_list operator()(variable_list&& inputs);  // Execute backward

 protected:
  virtual variable_list apply(variable_list&& inputs) = 0;  // Implement in subclass
};
```

**Key Methods**:

- **`operator()`**: Public interface that wraps `apply()` with profiling, anomaly detection, and hook execution
- **`apply()`**: Pure virtual method that subclasses implement with the actual backward logic
- **`next_edge(index)`**: Get the edge connecting output `index` to the next node
- **`add_input_metadata()`**: Record shape/dtype information for gradient validation
- **`should_compute_output(index)`**: Check if output `index` is needed (for optimization)

**Sequence Number** (lines 334-358):
- Thread-local counter that increases monotonically
- Used for two purposes:
  1. **Priority in backward engine**: Higher sequence number = executed first (reverse of forward order)
  2. **Profiler correlation**: Pair with `thread_id` to match backward ops with forward ops

**Topological Number** (lines 360-396):
- Length of longest path from this node to any leaf
- Leaf nodes (AccumulateGrad) have `topological_nr_ = 0`
- Allows O(1) pruning: if `topo_nr(X) <= topo_nr(Y)`, then no path exists from X to Y
- Must not change after node is added as a parent

### AccumulateGrad Node

**File**: Referenced in [torch/csrc/autograd/functions/basic_ops.h](../reference/pytorch/torch/csrc/autograd/functions/basic_ops.h)

Special `Node` subclass that accumulates gradients into a leaf tensor's `.grad` field.

```cpp
struct AccumulateGrad : public Node {
  explicit AccumulateGrad(Variable variable_);

  variable_list apply(variable_list&& grads) override {
    // Check versions to detect in-place modifications
    check_inplace_operations();

    // Accumulate gradient into variable.grad
    if (variable.grad().defined()) {
      variable.grad() += grads[0];
    } else {
      variable.grad() = grads[0];
    }

    return {};  // Sink node: no further outputs
  }

  Variable variable;  // The leaf tensor we're accumulating into
};
```

**Purpose**:
- Sink nodes in the autograd graph
- Receive gradients from upstream and write them into `Tensor.grad`
- Handle gradient accumulation across multiple backward passes
- Special priority: `sequence_nr = UINT64_MAX` (always executed first)

### SavedVariable

**File**: [torch/csrc/autograd/saved_variable.h](../reference/pytorch/torch/csrc/autograd/saved_variable.h#L22-L139)

Captures a snapshot of a tensor at a specific point in time for use during backward.

```cpp
class SavedVariable {
  at::Tensor data_;                           // Tensor data or tensor_data()
  std::shared_ptr<ForwardGrad> fw_grad_;      // Forward AD gradients
  std::weak_ptr<Node> weak_grad_fn_;          // For inplace views

  uint32_t saved_version_;                    // Version when saved
  uint32_t output_nr_;
  bool was_default_constructed_;
  bool is_inplace_on_view_;
  bool saved_original_;                       // Did we save full Variable?
  bool is_leaf_;
  bool is_output_;

  std::unique_ptr<SavedVariableHooks> hooks_; // Pack/unpack hooks
  std::shared_ptr<Node> grad_fn_;
  std::shared_ptr<Node> grad_accumulator_;
  bool requires_grad_;

 public:
  SavedVariable(const Variable& variable, bool is_output, bool is_inplace_on_view = false);
  Variable unpack(std::shared_ptr<Node> saved_for = nullptr) const;
  void register_hooks(std::unique_ptr<SavedVariableHooks>&& hooks);
  void reset_data();
};
```

**Two Storage Modes**:

1. **Direct storage** (`saved_original_ = true`):
   - Save the full `Variable` with all metadata
   - Used when saving won't create circular references
   - Can reconstruct without `grad_fn`

2. **Metadata-only storage** (`saved_original_ = false`):
   - Save only `variable.tensor_data()` (data without autograd info)
   - Store metadata separately (shape, dtype, requires_grad, etc.)
   - Used when saving the full Variable would create circular references
   - Requires passing `grad_fn` to `unpack()` to reconstruct

**Version Tracking**:
- Saves `version_counter_.current_version()` at construction
- Validates version matches during `unpack()` to detect illegal in-place operations
- Example error: modifying a tensor in-place after it's been saved for backward

**Hooks**:
- `pack_hook`: Called when saving, can customize what/how to save (e.g., move to CPU)
- `unpack_hook`: Called when unpacking, can customize reconstruction (e.g., move back to GPU)
- Used for checkpointing, offloading, and custom gradient behavior

## Gradient Edge Semantics

Every tensor has a **gradient edge** that determines how it participates in backprop:

```
Leaf Tensor:
  gradient_edge = Edge(grad_accumulator, 0)

Interior Tensor:
  gradient_edge = Edge(grad_fn, output_nr)
```

**Accessing Gradient Edge**:

```cpp
// From variable.h
Edge gradient_edge(const Variable& self) {
  auto meta = impl::get_autograd_meta(self);
  if (!meta) {
    return Edge();  // Not differentiable
  }

  if (meta->grad_fn_) {
    // Interior node
    return Edge(meta->grad_fn_, meta->output_nr_);
  } else {
    // Leaf node
    return Edge(impl::grad_accumulator(self), 0);
  }
}
```

## Differentiable Type Restriction

**File**: [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h#L44-L50)

```cpp
inline bool isDifferentiableType(at::ScalarType t) {
  return isFloatingType(t) || isComplexType(t);
}
```

Only floating-point and complex dtypes can have `requires_grad=True`:
- **Allowed**: Float32, Float64, Float16, BFloat16, Complex64, Complex128
- **Disallowed**: Int32, Int64, Bool, etc.

**Rationale**: Gradients are continuous values; integer tensors can't represent gradient information.

## View Tracking

### DifferentiableViewMeta

**File**: [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h#L723-L816)

For tensors that are views of other tensors, PyTorch uses a special `DifferentiableViewMeta` (subclass of `AutogradMeta`) to track the view relationship.

```cpp
struct DifferentiableViewMeta : public AutogradMeta {
 private:
  std::optional<ViewInfo> backward_info_;  // Backward mode view info
  std::optional<ViewInfo> forward_info_;   // Forward mode view info
  bool shared_view_info_;                  // Are forward/backward info the same?

  uint32_t attr_version_;                  // Version when grad_fn was created
  CreationMeta creation_meta_;             // How was this view created?

 public:
  bool has_bw_view() const { return backward_info_.has_value(); }
  bool has_fw_view() const { return shared_view_info_ || forward_info_.has_value(); }

  const ViewInfo& get_backward_view() const;
  const ViewInfo& get_forward_view() const;
};

struct ViewInfo {
  Variable base_;                          // The base tensor
  std::unique_ptr<ViewFunc> view_fn_;      // How to reconstruct view from base
  std::function<Variable(const Variable&)> rev_view_fn_;  // Reverse: view -> base
};
```

**Purpose**:
- Track base tensor for views (e.g., `y = x[1:5]`, `y` is a view of `x`)
- Enable gradient flow through view operations
- Detect illegal in-place operations on views
- Handle complex cases like in-place ops on views or bases

**View Tracking Example**:
```python
base = torch.randn(10, requires_grad=True)
view = base[2:7]                  # Create view

# view.autograd_meta is DifferentiableViewMeta
# view._base is base
# When base is modified in-place, view's grad_fn is updated (rebasing)
```

## AutogradState (Thread-Local)

**File**: [c10/core/AutogradState.h](../reference/pytorch/c10/core/AutogradState.h#L11-L83)

Thread-local state that controls autograd behavior:

```cpp
struct AutogradState {
  bool grad_mode_;                 // torch.no_grad() / torch.enable_grad()
  bool inference_mode_;            // torch.inference_mode()
  bool fw_grad_mode_;              // Forward-mode AD enabled?
  bool multithreading_enabled_;
  bool view_replay_enabled_;

  std::optional<SafePyObject> graph_exec_group_;

  static AutogradState& get_tls_state();
  static void set_tls_state(AutogradState state);
};
```

**grad_mode_**:
- Controls whether operations build the autograd graph
- `torch.no_grad()`: Temporarily disable gradient tracking
- `torch.enable_grad()`: Re-enable gradient tracking
- Context managers set/restore this flag

**inference_mode_**:
- More aggressive optimization than `no_grad()`
- Completely disables autograd machinery
- Can't re-enable gradients inside inference mode
- Results in faster execution and lower memory

## Data Flow Example

```python
import torch

# Step 1: Create leaf tensors
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# x.autograd_meta.requires_grad_ = True
# x.autograd_meta.grad_accumulator_ = AccumulateGrad(x)

# Step 2: Forward pass - build computational graph
z = x * y           # Creates MulBackward0 node
                    # z.grad_fn = MulBackward0
                    # MulBackward0.next_edges = [Edge(AccumulateGrad(x), 0),
                    #                            Edge(AccumulateGrad(y), 0)]

w = z + 1.0         # Creates AddBackward0 node
                    # w.grad_fn = AddBackward0
                    # AddBackward0.next_edges = [Edge(MulBackward0, 0)]

loss = w.sum()      # Creates SumBackward0 node

# Step 3: Backward pass
loss.backward()
# Engine starts from SumBackward0
# Calls SumBackward0.apply() -> gradient is 1.0
# Follows next_edges to AddBackward0
# Calls AddBackward0.apply() -> gradient still 1.0
# Follows next_edges to MulBackward0
# Calls MulBackward0.apply() -> splits gradient
#   grad_x = grad_output * y = 1.0 * 3.0 = 3.0
#   grad_y = grad_output * x = 1.0 * 2.0 = 2.0
# Follows next_edges to AccumulateGrad nodes
# AccumulateGrad(x).apply() writes 3.0 into x.grad
# AccumulateGrad(y).apply() writes 2.0 into y.grad

print(x.grad)  # tensor([3.])
print(y.grad)  # tensor([2.])
```

## Graph Construction

```
Forward Execution               Autograd Graph Built
─────────────────               ────────────────────

x = tensor([2.0],               x ──┐
     requires_grad=True)            │ AccumulateGrad(x)
                                    │
y = tensor([3.0],               y ──┐ AccumulateGrad(y)
     requires_grad=True)            │
                                    │
z = x * y                       z ──┤ MulBackward0
                                    │   next_edges[0] → AccumulateGrad(x)
                                    │   next_edges[1] → AccumulateGrad(y)
                                    │
w = z + 1.0                     w ──┤ AddBackward0
                                    │   next_edges[0] → MulBackward0
                                    │
loss = w.sum()              loss ──┤ SumBackward0
                                    │   next_edges[0] → AddBackward0
                                    │
loss.backward()                 [Execute backward from SumBackward0]
```

## Memory Management

**Reference Counting**:
- `Node` objects are managed via `std::shared_ptr`
- Tensors hold strong references to their `grad_fn`
- Nodes hold strong references to next nodes via `next_edges`
- Graph is freed when last reference to the root goes away

**Circular Reference Avoidance**:
- `SavedVariable` uses `weak_ptr` for some cases
- `grad_accumulator_` is a `weak_ptr` to prevent leaf → AccumulateGrad → leaf cycle
- Proper reference management ensures no memory leaks

**Graph Lifecycle**:
```python
x = torch.randn(5, requires_grad=True)
y = x * 2           # Creates MulBackward0

# At this point:
# - y holds shared_ptr to MulBackward0
# - MulBackward0 holds edge to AccumulateGrad(x)
# - x holds weak_ptr to AccumulateGrad(x)

y.backward(torch.ones(5))  # Execute backward

# After backward, if keep_graph=False (default):
# - Graph is freed
# - MulBackward0 is destroyed
# - y.grad_fn becomes nullptr (if graph is freed)

# Can't call y.backward() again (error: graph has been freed)
```

**Retaining Graphs**:
```python
y.backward(torch.ones(5), retain_graph=True)
# Graph is NOT freed
# Can call backward multiple times
y.backward(torch.ones(5))  # Works, gradients accumulate
```

## MLX Porting Considerations

### MLX Autograd Architecture

MLX provides `mlx.grad()` and `mlx.value_and_grad()` for automatic differentiation:

```python
import mlx.core as mx

# PyTorch style
x = torch.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)

# MLX style
def f(x):
    return (x * x).sum()

grad_f = mx.grad(f)
print(grad_f(mx.array([2.0])))  # Returns gradient directly
```

### Key Differences

**1. Eager vs Functional**:
- **PyTorch**: Builds graph eagerly during forward pass, imperative API
- **MLX**: Functional API, gradient computed on-demand via `grad()` wrapper

**2. Graph Storage**:
- **PyTorch**: Graph stored in tensor metadata (AutogradMeta)
- **MLX**: Graph implicitly defined by function composition

**3. Accumulation**:
- **PyTorch**: `.grad` attribute accumulates across multiple `.backward()` calls
- **MLX**: Each `grad()` call computes fresh gradients, no accumulation

**4. In-Place Operations**:
- **PyTorch**: Carefully tracks in-place ops, version counters, view rebasing
- **MLX**: Immutable arrays, no in-place modification concerns

### Porting Strategy

**Option 1: Native MLX API**
```python
# Don't emulate PyTorch's imperative autograd
# Use MLX's functional gradient API directly
def loss_fn(params, inputs, targets):
    predictions = model(params, inputs)
    return loss(predictions, targets)

grad_fn = mx.grad(loss_fn)
grads = grad_fn(params, inputs, targets)
```

**Option 2: Compatibility Layer**

For PyTorch model compatibility, create a thin wrapper:

```python
class TorchCompatTensor:
    def __init__(self, data):
        self._data = data
        self._grad = None
        self._requires_grad = False
        self._grad_fn = None

    def backward(self, gradient=None):
        # Compute gradient using MLX
        if self._grad_fn is None:
            return

        # Use MLX grad() to compute gradients
        grads = self._grad_fn.backward(gradient)

        # Accumulate into .grad
        if self._grad is None:
            self._grad = grads
        else:
            self._grad = self._grad + grads
```

**Challenges**:

1. **Dynamic Graphs**: PyTorch's tape-based approach allows different graphs per iteration. MLX prefers static graphs for optimization. Solution: Re-trace per iteration or use compilation where possible.

2. **Hooks**: PyTorch has extensive pre/post hooks on tensors and functions. MLX doesn't have direct equivalents. Solution: Manual instrumentation points.

3. **Double Backward**: PyTorch supports higher-order gradients via `create_graph=True`. MLX supports this via nested `grad()` calls but syntax differs.

4. **Custom Gradients**: PyTorch uses `torch.autograd.Function` with `forward()` and `backward()`. MLX uses `mx.custom_function` with vjp/jvp. Different APIs.

5. **View Tracking**: PyTorch extensively tracks views for gradient correctness. MLX arrays are immutable, so views are copies. Less complex but different semantics.

### Mapping Table

| PyTorch Concept | MLX Equivalent | Notes |
|-----------------|----------------|-------|
| `requires_grad=True` | Function input to `grad()` | Functional vs attribute |
| `.backward()` | `grad_fn(x)` | Must define gradient function |
| `.grad` | Return value of `grad_fn` | No accumulation by default |
| `torch.autograd.grad()` | `mx.grad()` | Similar functional API |
| `torch.no_grad()` | Don't call `grad()` | Simpler in MLX |
| `.retain_grad()` | Not needed | All intermediates can be captured |
| `AccumulateGrad` | Manual accumulation | `grads = grads + new_grads` |
| `SavedVariable` | Captured in closure | Python closures sufficient |
| View tracking | Not needed | Immutable arrays |

### Example: Simple Porting

**PyTorch**:
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.pow(2).sum()
y.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

**MLX Direct**:
```python
import mlx.core as mx

def f(x):
    return (x ** 2).sum()

grad_f = mx.grad(f)
x = mx.array([1.0, 2.0, 3.0])
print(grad_f(x))  # array([2., 4., 6.])
```

**MLX with Compatibility Layer**:
```python
class Variable:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self._backward_fn = None

    def backward(self):
        if self._backward_fn:
            self.grad = self._backward_fn()

# Would require significant infrastructure to match PyTorch's API
```

## Critical File References

- [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h): AutogradMeta, DifferentiableViewMeta, ViewInfo
- [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h): Node base class, edge_list
- [torch/csrc/autograd/edge.h](../reference/pytorch/torch/csrc/autograd/edge.h): Edge structure
- [torch/csrc/autograd/saved_variable.h](../reference/pytorch/torch/csrc/autograd/saved_variable.h): SavedVariable snapshots
- [torch/csrc/autograd/engine.h](../reference/pytorch/torch/csrc/autograd/engine.h): Backward engine (next doc)
- [c10/core/AutogradState.h](../reference/pytorch/c10/core/AutogradState.h): Thread-local state
- [c10/core/VariableVersion.h](../reference/pytorch/c10/core/VariableVersion.h): Version tracking for in-place ops
- [torch/csrc/autograd/functions/basic_ops.h](../reference/pytorch/torch/csrc/autograd/functions/basic_ops.h): AccumulateGrad, GraphRoot

## Next Steps

Now that we understand the core autograd abstractions, the next documents will cover:

1. **computational-graph.md**: How nodes and edges form the DAG, graph construction, topological ordering
2. **backward-engine.md**: How the Engine executes the backward pass, scheduling, threading
3. **gradient-formulas.md**: How derivatives are defined in derivatives.yaml, common patterns
4. **custom-functions.md**: torch.autograd.Function API for custom gradients
