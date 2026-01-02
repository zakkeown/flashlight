# Computational Graph

## Purpose

PyTorch's autograd builds a **Directed Acyclic Graph (DAG)** during the forward pass to represent the computational flow. This graph encodes:
- Which operations were performed
- The dependencies between tensors
- The information needed to compute gradients during backward

The computational graph is:
- **Dynamic**: Built on-the-fly during forward execution, can change between iterations
- **Implicit**: Constructed automatically when operations are performed on tensors with `requires_grad=True`
- **Directed**: Edges point from outputs back to inputs (in the direction gradients flow)
- **Acyclic**: No cycles allowed (prevents infinite loops during backward)

This document explains how nodes and edges form the DAG, how the graph is constructed during forward execution, and how topological ordering enables efficient backward passes.

## Graph Structure

### Nodes and Edges

```
Computational Graph for z = (x + y) * w

Tensors (Values):          Gradient Functions (Nodes):

    x (leaf)                      AccumulateGrad(x)
    y (leaf)                             ▲
    w (leaf)                             │
                                   AddBackward0 ────┐
                               ┌─────┘     ▲       │
    x + y = temp              │            │       │
                              │            │       │
                              │      AccumulateGrad(y)
                              │
    temp * w = z              │
                              │            ▲
                         MulBackward0      │
                              ▲            │
                              │      AccumulateGrad(w)
                              │
                        SumBackward0 (if loss = z.sum())


Graph Structure (backward direction):

SumBackward0
    │ next_edges[0]
    ▼
MulBackward0
    │ next_edges[0]        │ next_edges[1]
    ▼                      ▼
AddBackward0          AccumulateGrad(w)
    │ next_edges[0]        │ next_edges[1]
    ▼                      ▼
AccumulateGrad(x)     AccumulateGrad(y)
```

**Key Insight**: The computational graph represents the backward pass, not the forward pass. Edges point in the direction gradients flow (from loss backward to inputs).

### Node Types

1. **Interior Nodes** (`Node` subclasses):
   - Represent differentiable operations (e.g., `AddBackward0`, `MulBackward0`)
   - Have `next_edges` pointing to upstream gradient functions
   - Implement `apply()` to compute gradients
   - Created during forward pass when operations are performed

2. **Leaf Nodes** (`AccumulateGrad`):
   - Special sink nodes that accumulate gradients into leaf tensors
   - Have no outgoing edges (`next_edges` is empty or points to nothing meaningful)
   - Write gradients into `Tensor.grad`
   - Always exist for leaf tensors with `requires_grad=True`

3. **Graph Root** (conceptually):
   - The node where `backward()` is called (typically the loss)
   - Receives initial gradient of 1.0
   - Starting point for backward traversal

### Edge Structure

**File**: [torch/csrc/autograd/edge.h](../reference/pytorch/torch/csrc/autograd/edge.h#L14-L39)

```cpp
struct Edge {
  std::shared_ptr<Node> function;  // Target gradient function
  uint32_t input_nr;                // Which input of the target

  Edge() noexcept : function(nullptr), input_nr(0) {}
  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept;
  bool is_valid() const noexcept { return function != nullptr; }
};
```

**Purpose of `input_nr`**:
- Many operations have multiple inputs (e.g., `add` has two tensors)
- During backward, gradients must be routed to the correct input
- `input_nr` identifies which input this edge corresponds to

**Example**:
```python
x = torch.tensor([1.0], requires_grad=True)  # input 0
y = torch.tensor([2.0], requires_grad=True)  # input 1
z = x + y  # AddBackward0

# z.grad_fn.next_edges has two edges:
#   Edge(AccumulateGrad(x), input_nr=0)  # gradient w.r.t. first input
#   Edge(AccumulateGrad(y), input_nr=1)  # gradient w.r.t. second input
```

### InputBuffer: Gradient Accumulation

**File**: [torch/csrc/autograd/input_buffer.h](../reference/pytorch/torch/csrc/autograd/input_buffer.h#L17-L54)

When multiple edges point to the same node (gradient contributions from different paths), gradients are accumulated in an `InputBuffer`:

```cpp
struct InputBuffer {
  std::vector<Variable> buffer;                      // Accumulated gradients
  std::vector<std::optional<c10::Stream>> opt_accum_streams;
  std::vector<std::optional<c10::Event>> ready_events;
  std::vector<std::optional<c10::Stream>> ready_streams;

  void add(
      size_t pos,
      Variable&& var,
      const std::optional<c10::Stream>& opt_producer_stream,
      const std::optional<c10::Stream>& opt_consumer_stream,
      Node* fn);
};
```

**Purpose**:
- Accumulate gradients from multiple sources before calling node's `apply()`
- Handle stream synchronization for GPU tensors
- Implement implicit addition at node inputs

**Example**:
```python
x = torch.tensor([1.0], requires_grad=True)
y = x + x  # x is used twice

# During backward:
# AddBackward0 receives gradient 1.0 from y
# AddBackward0.apply() splits: grad_x = 1.0 + 1.0 = 2.0
# (Technically, add's derivative is ∂(x+x)/∂x = 1 + 1 = 2)
```

## Graph Construction

### Forward Pass: Building the Graph

Graph construction happens automatically during the forward pass through operator hooks registered via the dispatcher.

**Mechanism** (simplified):

1. **Operation Execution**: User calls `z = x + y`
2. **Dispatcher Resolution**: Routes to the appropriate kernel (CPU, CUDA, etc.)
3. **Autograd Layer**: If any input has `requires_grad=True` and `GradMode::is_enabled()`:
   - Create gradient function node (e.g., `AddBackward0`)
   - Connect node to inputs via `next_edges`
   - Store necessary information for backward (via `SavedVariable`)
   - Set output's `grad_fn` to the new node
4. **Return Output**: Output tensor has `grad_fn` pointing to the operation

**Code Flow**:

```cpp
// Simplified from generated code (torch/csrc/autograd/generated/)

Tensor add_autograd(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // Check if we need to build the graph
  bool requires_grad = any_variable_requires_grad({self, other});

  if (!requires_grad || !GradMode::is_enabled()) {
    // Fast path: no autograd
    return at::add(self, other, alpha);
  }

  // Create gradient function
  auto grad_fn = std::make_shared<AddBackward0>();

  // Set up edges to inputs
  grad_fn->set_next_edges(collect_next_edges(self, other));

  // Save values needed for backward
  grad_fn->save_for_backward({self, other, alpha});

  // Execute forward operation
  auto result = at::add(self, other, alpha);

  // Connect result to grad_fn
  set_history(result, grad_fn);

  return result;
}
```

### Lazy Node Creation

Gradient function nodes are created **lazily** only when needed:
- Not created if no inputs require gradients
- Not created if `torch.no_grad()` is active
- Not created for non-differentiable operations

**Performance Benefit**: Inference mode doesn't pay autograd overhead.

### Multi-Output Operations

Some operations produce multiple outputs. Each output can have a different `output_nr`:

```python
x = torch.randn(10, requires_grad=True)
values, indices = torch.topk(x, 5)

# Both values and indices come from the same TopKBackward0 node
# values.grad_fn == TopKBackward0, values.output_nr == 0
# indices is non-differentiable, so indices.grad_fn == None
```

**Implementation**:
```cpp
// For multi-output functions
auto grad_fn = std::make_shared<TopKBackward0>();

// Output 0: values (differentiable)
set_history(values, grad_fn, /*output_nr=*/0);

// Output 1: indices (non-differentiable, no grad_fn)
```

### Detachment and Views

**Detaching**:
```python
x = torch.randn(5, requires_grad=True)
y = x * 2
z = y.detach()  # z has no grad_fn, breaks graph

# Gradient flow stops at z
# z.backward() would error (no grad_fn)
```

**Views** (covered in detail in autograd-overview.md):
```python
x = torch.randn(10, requires_grad=True)
y = x[2:7]  # View of x

# y is a DifferentiableView
# y.grad_fn is a special view node
# Gradients flow through views back to base
```

## Topological Ordering

### Topological Number

**File**: [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h#L360-L396)

Each `Node` has a `topological_nr_` representing the longest path from this node to any leaf:

```cpp
// From Node class
uint64_t topological_nr_ = 0;
mutable bool has_parent_ = false;

uint64_t topological_nr() const noexcept {
  has_parent_ = true;
  return topological_nr_;
}

void update_topological_nr(const Edge& edge) {
  Node* node = edge.function.get();
  if (node) {
    auto topo_nr = node->topological_nr();
    if (topological_nr_ <= topo_nr) {
      topological_nr_ = topo_nr + 1;
    }
  }
}
```

**Invariants**:
- Leaf nodes (AccumulateGrad): `topological_nr_ = 0`
- Interior nodes: `topological_nr_ = max(child_topo_nr) + 1`
- **Property**: If path exists from X to Y, then `topo_nr(X) > topo_nr(Y)`
- **Contrapositive**: If `topo_nr(X) <= topo_nr(Y)`, then NO path exists from X to Y

**Purpose**:
1. **Graph Pruning**: During backward, skip nodes that can't reach the desired outputs
2. **Execution Order**: Engine can prioritize nodes with higher topological numbers
3. **Cycle Detection**: Helps ensure the graph remains acyclic

**Example**:
```
z = (x + y) * w

AccumulateGrad(x): topo_nr = 0
AccumulateGrad(y): topo_nr = 0
AccumulateGrad(w): topo_nr = 0
AddBackward0:      topo_nr = 1  (max(0, 0) + 1)
MulBackward0:      topo_nr = 2  (max(1, 0) + 1)
```

### Sequence Number

**File**: [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h#L334-L358)

Each `Node` has a `sequence_nr_` representing when it was created:

```cpp
// Thread-local counter
uint64_t sequence_nr_;

explicit Node(uint64_t sequence_nr, edge_list&& next_edges = edge_list())
    : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {
  // ...
}

explicit Node(edge_list&& next_edges = edge_list())
    : Node(
          /*sequence_nr=*/at::sequence_number::get_and_increment(),
          std::move(next_edges)) {}
```

**Uses**:

1. **Execution Priority**: Higher sequence number = created later = executed first during backward
   - Ensures backward runs in reverse order of forward
   - Nodes from later in forward pass have higher priority

2. **Profiler Correlation**: Pair (sequence_nr, thread_id) uniquely identifies a node
   - Helps match backward operations to forward operations in profiler output
   - Thread-local counter means sequence numbers are only comparable within a thread

**Special Case: AccumulateGrad**:
```cpp
AccumulateGrad::AccumulateGrad(Variable variable_)
    : Node(/*sequence_nr=*/UINT64_MAX), variable(std::move(variable_)) {
  // UINT64_MAX ensures AccumulateGrad is always executed first
}
```

AccumulateGrad always has maximum priority to ensure gradients are written to leaf tensors as soon as possible.

## GraphTask Metadata

**File**: [torch/csrc/autograd/graph_task.h](../reference/pytorch/torch/csrc/autograd/graph_task.h#L18-L206)

A `GraphTask` represents a single execution of `backward()` and holds all metadata for that execution:

```cpp
struct GraphTask : std::enable_shared_from_this<GraphTask> {
  std::atomic<uint64_t> outstanding_tasks_{0};  // How many nodes left to execute
  std::atomic_bool has_error_{false};
  std::atomic_bool future_completed_{false};
  bool keep_graph_;                              // Retain graph after backward?

  std::mutex mutex_;
  std::unordered_map<Node*, InputBuffer> not_ready_;  // Nodes waiting for inputs
  std::unordered_map<Node*, int> dependencies_;       // Dependency counts

  std::unordered_set<Node*> nodes_in_graph_;     // All nodes reachable from roots
  c10::SmallVector<Node*, 4> graph_roots_;       // Where backward starts

  // Execution info for selective backward (when inputs= is specified)
  struct ExecInfo {
    bool needed_ = false;                        // Should this node execute?
    std::unique_ptr<std::vector<Capture>> captures_;  // Capture gradients?

    bool should_execute() const { return needed_ || captures_; }
  };
  std::unordered_map<Node*, ExecInfo> exec_info_;

  std::vector<Variable> captured_vars_;          // Gradients captured for user
  at::ThreadLocalState thread_locals_;           // Thread-local state snapshot
  std::unordered_set<c10::Stream> leaf_streams;  // Streams used by leaf ops

  int owner_{NO_DEVICE};                         // Device that owns this task
  const int reentrant_depth_;                    // Nesting level of backward calls

  std::shared_ptr<ReadyQueue> cpu_ready_queue_;  // Queue for CPU tasks
  c10::intrusive_ptr<at::ivalue::Future> future_result_;  // Async completion

  std::vector<std::function<void()>> final_callbacks_;
  utils::DelayWarningHandler warning_handler_;
  uint64_t id_;
};
```

### Dependency Tracking

**File**: [torch/csrc/autograd/graph_task.h](../reference/pytorch/torch/csrc/autograd/graph_task.h#L30-L31)

Before backward execution, the engine computes dependencies:

```cpp
std::unordered_map<Node*, int> dependencies_;       // How many inputs needed?
std::unordered_map<Node*, InputBuffer> not_ready_;  // Accumulated gradients so far
```

**Algorithm** (simplified):

```cpp
void Engine::compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr) {
  std::queue<Node*> queue;
  std::unordered_set<Node*> seen;

  queue.push(root);
  seen.insert(root);

  while (!queue.empty()) {
    Node* node = queue.front();
    queue.pop();

    for (const Edge& edge : node->next_edges()) {
      if (!edge.is_valid()) continue;

      Node* next_fn = edge.function.get();

      // Skip if node is below minimum topological number (pruning)
      if (next_fn->topological_nr() < min_topo_nr) continue;

      // Increment dependency count
      task.dependencies_[next_fn]++;

      // Add to graph
      task.nodes_in_graph_.insert(next_fn);

      // Continue traversal
      if (seen.find(next_fn) == seen.end()) {
        queue.push(next_fn);
        seen.insert(next_fn);
      }
    }
  }
}
```

**Purpose**:
- `dependencies_[node]` tracks how many inputs this node needs before it can execute
- When a gradient arrives, decrement the count
- When count reaches 0, node is ready to execute

### Exec Info: Selective Backward

**File**: [torch/csrc/autograd/graph_task.h](../reference/pytorch/torch/csrc/autograd/graph_task.h#L47-L115)

The `exec_info_` map enables **selective backward** when `inputs=` parameter is specified:

```python
# Compute gradients only w.r.t. specific inputs
x = torch.randn(5, requires_grad=True)
y = torch.randn(5, requires_grad=True)
z = torch.randn(5, requires_grad=True)

out = x * y + z
grads = torch.autograd.grad(out, [x], grad_outputs=torch.ones_like(out))
# Only computes gradient for x, not y or z
# Engine skips nodes that don't contribute to x's gradient
```

**Implementation**:

```cpp
struct ExecInfo {
  bool needed_ = false;  // Is this node needed for the requested inputs?
  std::unique_ptr<std::vector<Capture>> captures_;  // Should we capture output?

  bool should_execute() const {
    return needed_ || captures_;
  }
};
```

**Algorithm**:
1. Start from requested `inputs` and mark them as `needed_`
2. Backward traverse graph, marking nodes as needed if they can reach a needed leaf
3. During backward, skip nodes where `exec_info_[node].should_execute() == false`

**Default Behavior**:
- If `exec_info_` is empty (normal `.backward()`), all nodes execute
- If `exec_info_` is non-empty (`.grad()` or `.backward(inputs=...)`), only marked nodes execute

## Graph Lifetime

### Reference Counting

Nodes are managed via `std::shared_ptr`:

```
Tensor (y) ──shared_ptr──> Node (MulBackward0)
                              │
                              │ next_edges[0]
                              ▼
                           Node (AddBackward0)
                              │
                              │ next_edges[0]
                              ▼
                           AccumulateGrad(x)
                              │
                              │ weak_ptr (to avoid cycle)
                              ▼
                           Tensor (x)
```

**Circular Reference Prevention**:
- Leaf tensors hold **weak_ptr** to their `AccumulateGrad` node
- This prevents: Tensor → AccumulateGrad → Tensor cycle
- `AccumulateGrad` is created on-demand via `grad_accumulator()`

### Graph Destruction

**After backward()**:

```python
x = torch.randn(5, requires_grad=True)
y = x * 2
z = y + 1

z.backward(torch.ones(5))
# After backward, if keep_graph=False (default):
# - z, y still exist as tensors
# - z.grad_fn and y.grad_fn are freed
# - Graph is destroyed

# Can't call backward again
z.backward(torch.ones(5))  # ERROR: graph has been freed
```

**Retaining Graph**:

```python
z.backward(torch.ones(5), retain_graph=True)
# Graph is kept alive
# Can call backward multiple times
z.backward(torch.ones(5))  # OK, gradients accumulate
```

**Implementation**:

```cpp
void GraphTask::mark_as_completed_and_run_post_processing() {
  if (!keep_graph_) {
    // Release all SavedVariable data
    for (Node* node : nodes_in_graph_) {
      node->release_variables();
    }
  }
  // ... other cleanup
}
```

## Advanced Graph Patterns

### In-Place Operations

In-place operations modify tensors and require special handling:

```python
x = torch.randn(5, requires_grad=True)
y = x + 1
y.add_(2)  # In-place operation

# Problem: If backward is called later, we need original y values
# Solution: Version tracking via VariableVersion
```

**Version Tracking**:
- Each tensor has a `version_counter_`
- In-place ops increment the version
- `SavedVariable` stores the version when saved
- During backward, check if version matches
- If mismatch, error (tensor was modified in-place after being saved)

**Example Error**:
```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
x.add_(1)  # Modifies x in-place
y.backward(torch.ones(5))  # ERROR: x was modified in-place
```

### Reentrant Backward

Calling `backward()` inside a backward hook creates reentrant execution:

```python
def hook(grad):
    # Calling backward inside a hook
    some_tensor.backward()
    return grad

x = torch.randn(5, requires_grad=True)
x.register_hook(hook)
```

**Challenges**:
- Need to handle nested `GraphTask` execution
- Thread-local state must be saved/restored
- Potential for deadlocks if not carefully managed

**Solution** (simplified):
```cpp
struct GraphTask {
  int owner_{NO_DEVICE};         // Which device/thread owns this?
  const int reentrant_depth_;    // How many levels deep?

  // Max depth before spawning new thread
  static constexpr int MAX_DEPTH = 60;
};

// In Engine::execute()
if (task->reentrant_depth_ >= MAX_DEPTH) {
  // Spawn new thread to avoid stack overflow
  std::thread([task]() {
    execute_graph_task(task);
  }).detach();
}
```

### Double Backward (Higher-Order Gradients)

Computing gradients of gradients requires `create_graph=True`:

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2).sum()

# First backward: build graph for gradients
grad_x = torch.autograd.grad(y, x, create_graph=True)[0]

# grad_x has its own grad_fn
# Second backward: gradient of gradient
grad_grad_x = torch.autograd.grad(grad_x.sum(), x)[0]
```

**Implementation**:
- When `create_graph=True`, backward operations create their own `Node`s
- Gradient computation becomes part of the forward pass for the next level
- Can nest arbitrarily deep (limited by memory/stack)

**Graph Structure**:
```
First-order graph:
  x (requires_grad=True) → y → backward → grad_x

Second-order graph (created by create_graph=True):
  x (requires_grad=True) → grad_x (has grad_fn) → backward → grad_grad_x
```

## Graph Visualization Example

**Complete Example**:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = x * y           # MulBackward0
w = z.pow(2)        # PowBackward0
loss = w.sum()      # SumBackward0

loss.backward()
```

**Graph Structure**:

```
Forward pass (data flow):
  x(2.0) ──┐
           ├─> mul ──> z(6.0) ──> pow ──> w(36.0) ──> sum ──> loss(36.0)
  y(3.0) ──┘

Backward graph (gradient flow):

    SumBackward0  (sequence_nr=N+2, topo_nr=3)
         │
         │ next_edges[0]: Edge(PowBackward0, input_nr=0)
         ▼
    PowBackward0  (sequence_nr=N+1, topo_nr=2)
         │
         │ next_edges[0]: Edge(MulBackward0, input_nr=0)
         ▼
    MulBackward0  (sequence_nr=N, topo_nr=1)
         │
         ├─> next_edges[0]: Edge(AccumulateGrad(x), input_nr=0)
         │
         └─> next_edges[1]: Edge(AccumulateGrad(y), input_nr=0)
         │                   │
         ▼                   ▼
    AccumulateGrad(x)   AccumulateGrad(y)
    (sequence_nr=UINT64_MAX, topo_nr=0)

Backward execution (in sequence_nr order):
  1. SumBackward0: receives grad=1.0, outputs grad=1.0
  2. PowBackward0: receives grad=1.0, computes grad=2*z=12.0
  3. MulBackward0: receives grad=12.0, computes:
     - grad_x = 12.0 * y = 36.0
     - grad_y = 12.0 * x = 24.0
  4. AccumulateGrad(x): receives grad=36.0, writes to x.grad
  5. AccumulateGrad(y): receives grad=24.0, writes to y.grad

Result:
  x.grad = 36.0  (derivative: ∂loss/∂x = 2*x*y^2)
  y.grad = 24.0  (derivative: ∂loss/∂y = 2*x^2*y)
```

## MLX Porting Considerations

### MLX Graph Representation

MLX uses a different approach:

**PyTorch**: Explicit graph stored in tensor metadata
```python
# Graph built during forward
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
# y.grad_fn points to MulBackward0
# MulBackward0.next_edges points to AccumulateGrad(x)
```

**MLX**: Implicit graph defined by function composition
```python
import mlx.core as mx

def f(x):
    return x * 2

# Graph is implicit in function definition
# mx.grad(f) analyzes function to compute gradient
grad_f = mx.grad(f)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Graph Storage** | Explicit (AutogradMeta, Node objects) | Implicit (function traces) |
| **Construction** | Eager (during forward) | Lazy (when grad() called) |
| **Memory Overhead** | Every intermediate has metadata | Minimal overhead |
| **Flexibility** | Different graph per iteration | Easier to optimize static graphs |
| **Debugging** | Can inspect `.grad_fn` | Inspect compiled primitives |

### Porting Challenges

**1. Dynamic Graphs**:

PyTorch's eager graph construction allows control flow:
```python
# Different graphs for different inputs
def forward(x, use_dropout):
    if use_dropout:
        x = F.dropout(x, p=0.5)
    return x
```

MLX equivalent requires conditional primitives or recompilation:
```python
# Need to handle control flow explicitly
def forward(x, use_dropout):
    return mx.where(use_dropout, dropout(x), x)
```

**2. In-Place Operations**:

PyTorch tracks versions; MLX arrays are immutable:
```python
# PyTorch
x = torch.randn(5, requires_grad=True)
x.add_(1)  # In-place, version increments

# MLX
x = mx.array([1.0])
x = x + 1  # Creates new array, x is reassigned
```

**3. Graph Inspection**:

PyTorch allows introspection:
```python
# Can walk the graph
node = y.grad_fn
while node is not None:
    print(node)
    node = node.next_functions[0][0] if node.next_functions else None
```

MLX doesn't expose graph structure directly. Use compilation/tracing tools instead.

**4. Selective Backward**:

PyTorch's `exec_info` enables fine-grained control:
```python
torch.autograd.grad(output, [input_subset])  # Only compute needed gradients
```

MLX requires defining separate functions:
```python
# Define gradient function for specific inputs
grad_fn = mx.grad(f, argnums=[0])  # Only gradient w.r.t. first argument
```

### Recommended MLX Approach

Instead of emulating PyTorch's graph structure:

**Option 1: Pure Functional**
```python
# Embrace MLX's functional style
def model_fn(params, inputs):
    return forward_pass(params, inputs)

def loss_fn(params, inputs, targets):
    predictions = model_fn(params, inputs)
    return compute_loss(predictions, targets)

# Compute gradients functionally
grad_fn = mx.grad(loss_fn)
grads = grad_fn(params, inputs, targets)

# Update parameters
params = {k: v - lr * grads[k] for k, v in params.items()}
```

**Option 2: Compatibility Layer for Model Porting**
```python
class TorchCompatTensor:
    """Minimal compatibility for loading PyTorch models"""
    def __init__(self, data):
        self.data = data
        self._grad_fn = None  # Store for compatibility, but don't build graph

    def backward(self):
        # Use MLX grad() instead of graph traversal
        raise NotImplementedError("Use mx.grad() instead")
```

### Performance Implications

**PyTorch Graph Overhead**:
- Every operation allocates a `Node`
- Stores `SavedVariable` for backward
- Reference counting overhead
- Good for dynamic models, some overhead for static models

**MLX Advantages**:
- Lazy evaluation allows graph optimization
- Fusion opportunities (combine multiple ops)
- Lower memory footprint for static graphs
- Better suited for deployment

**Trade-offs**:
- PyTorch: Easier debugging, more flexible, higher overhead
- MLX: Better performance, harder debugging, less flexible

## Critical File References

- [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h): Node base class, topological/sequence numbering
- [torch/csrc/autograd/edge.h](../reference/pytorch/torch/csrc/autograd/edge.h): Edge structure connecting nodes
- [torch/csrc/autograd/variable.h](../reference/pytorch/torch/csrc/autograd/variable.h): AutogradMeta, gradient_edge()
- [torch/csrc/autograd/graph_task.h](../reference/pytorch/torch/csrc/autograd/graph_task.h): GraphTask, dependency tracking, exec_info
- [torch/csrc/autograd/input_buffer.h](../reference/pytorch/torch/csrc/autograd/input_buffer.h): InputBuffer for gradient accumulation
- [torch/csrc/autograd/functions/accumulate_grad.h](../reference/pytorch/torch/csrc/autograd/functions/accumulate_grad.h): AccumulateGrad implementation
- [torch/csrc/autograd/engine.cpp](../reference/pytorch/torch/csrc/autograd/engine.cpp): Graph traversal and execution (next doc)
- [c10/core/VariableVersion.h](../reference/pytorch/c10/core/VariableVersion.h): Version tracking for in-place ops

## Next Steps

This document covered the computational graph structure. The next documents will cover:

1. **backward-engine.md**: How the Engine executes backward pass, scheduling, threading, ready queue
2. **gradient-formulas.md**: How derivatives are defined, derivatives.yaml, common patterns
3. **custom-functions.md**: torch.autograd.Function API for custom backward passes
