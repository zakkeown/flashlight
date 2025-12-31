# Backward Engine

## Purpose

The **Engine** is PyTorch's backward execution system that traverses the computational graph in reverse topological order to compute gradients. It manages:
- Scheduling and executing gradient functions (Nodes)
- Dependency tracking to determine when nodes are ready
- Multi-device execution with proper stream synchronization
- Reentrant backward calls (calling backward inside backward)
- Thread pool management for concurrent execution

The Engine orchestrates the backward pass, transforming the declarative computational graph into an imperative execution plan.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Backward Engine                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User calls loss.backward()                                     │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Engine::execute()                                       │  │
│  │  - Create GraphTask                                      │  │
│  │  - Compute dependencies                                  │  │
│  │  - Initialize ReadyQueue with root                       │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  thread_main() - Worker Thread Loop                      │  │
│  │                                                           │  │
│  │  while (graph_task not complete):                        │  │
│  │    1. Pop NodeTask from ReadyQueue (priority queue)      │  │
│  │    2. evaluate_function()                                │  │
│  │    3. Decrement outstanding_tasks                        │  │
│  │    4. If dependencies met, push next nodes to queue      │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  evaluate_function()                                     │  │
│  │  - Stream synchronization                                │  │
│  │  - Check exec_info (selective backward)                  │  │
│  │  - call_function()                                       │  │
│  │  - Distribute outputs to next nodes                      │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  call_function()                                         │  │
│  │  - Pre-hooks                                             │  │
│  │  - Node::operator() (executes apply())                   │  │
│  │  - Post-hooks                                            │  │
│  │  - Validate outputs                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────┐     ┌───────────────────┐               │
│  │  CPU ReadyQueue   │     │ Device ReadyQueues│               │
│  │  (per GraphTask)  │     │ (shared, CUDA/XLA)│               │
│  │                   │     │                   │               │
│  │  Priority Queue   │     │  Priority Queue   │               │
│  │  by sequence_nr   │     │  by sequence_nr   │               │
│  └───────────────────┘     └───────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Engine

**File**: [torch/csrc/autograd/engine.h](../reference/pytorch/torch/csrc/autograd/engine.h#L130-L283)

The singleton Engine manages backward execution across all devices:

```cpp
struct Engine {
  // Singleton access
  static Engine& get_default_engine();

  // Main entry point for backward
  virtual variable_list execute(
      const edge_list& roots,          // Starting nodes (usually loss)
      const variable_list& inputs,     // Initial gradients
      bool keep_graph,                 // Retain graph after backward?
      bool create_graph,               // Build graph for double backward?
      bool accumulate_grad,            // Accumulate into .grad?
      const edge_list& outputs = {}    // Selective backward targets
  );

  // Worker thread management
  std::vector<std::shared_ptr<ReadyQueue>> device_ready_queues_;
  void start_device_threads();
  void thread_init(int device, const std::shared_ptr<ReadyQueue>& ready_queue);
  void thread_main(const std::shared_ptr<GraphTask>& graph_task);

  // Function evaluation
  void evaluate_function(
      std::shared_ptr<GraphTask>& graph_task,
      Node* func,
      InputBuffer& inputs,
      const std::shared_ptr<ReadyQueue>& cpu_ready_queue);

  // Dependency computation
  void compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr);

  // Thread pool for reentrant backwards
  std::shared_ptr<ThreadPoolShared> thread_pool_shared_;
  int max_recursion_depth_{60};  // MAX_DEPTH before spawning new thread

 private:
  std::atomic<uint32_t> non_reentrant_device_thread_count_;
  std::vector<std::function<void()>> final_callbacks_;
  std::mutex post_callbacks_lock_;
};
```

### ReadyQueue

**File**: [torch/csrc/autograd/engine.h](../reference/pytorch/torch/csrc/autograd/engine.h#L86-L125)

Priority queue that orders nodes for execution:

```cpp
struct ReadyQueue {
 private:
  struct CompareNodeTaskTime {
    bool operator()(NodeTask const& t1, NodeTask const& t2) {
      // Shutdown tasks have highest priority
      if (t2.isShutdownTask_) return true;
      if (t1.isShutdownTask_) return false;

      // Empty tasks next
      if (!t1.fn_) return false;
      if (!t2.fn_) return true;

      // Then by reentrant depth (deeper first)
      if (t1.getReentrantDepth() != t2.getReentrantDepth()) {
        return t1.getReentrantDepth() < t2.getReentrantDepth();
      }

      // Finally by sequence_nr (higher first = later in forward = earlier in backward)
      return t1.fn_->sequence_nr() < t2.fn_->sequence_nr();
    }
  };

  std::priority_queue<NodeTask, std::vector<NodeTask>, CompareNodeTaskTime> heap_;
  std::condition_variable not_empty_;
  mutable std::mutex mutex_;

 public:
  void push(NodeTask item, bool incrementOutstandingTasks = true);
  NodeTask pop();  // Blocks until task available
  bool empty() const;
  size_t size() const;
};
```

**Priority Order**:
1. Shutdown tasks (highest priority)
2. Error tasks (nodes with invalid GraphTask)
3. Deeper reentrant backwards
4. Higher sequence_nr (executed later in forward → executed first in backward)

### NodeTask

**File**: [torch/csrc/autograd/engine.h](../reference/pytorch/torch/csrc/autograd/engine.h#L51-L73)

Represents a unit of work for the engine:

```cpp
struct NodeTask {
  std::weak_ptr<GraphTask> base_;  // Which backward this belongs to
  std::shared_ptr<Node> fn_;        // The gradient function to execute
  InputBuffer inputs_;              // Accumulated gradients for this node
  bool isShutdownTask_;             // Special task to shutdown threads

  int getReentrantDepth() const {
    std::shared_ptr<GraphTask> graph_task = base_.lock();
    if (graph_task) {
      return graph_task->reentrant_depth_;
    } else {
      // Graph task invalid (error), move to front of queue
      return std::numeric_limits<int>::max();
    }
  }
};
```

## Execution Flow

### 1. Engine::execute()

Entry point when user calls `loss.backward()`:

```cpp
variable_list Engine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {

  // 1. Create GraphTask
  auto graph_task = std::make_shared<GraphTask>(
      keep_graph,
      create_graph,
      /*reentrant_depth=*/current_depth,
      cpu_ready_queue,
      graph_roots);

  // 2. Initialize execution info (for selective backward)
  if (!outputs.empty()) {
    graph_task->init_to_execute(
        *graph_root,
        outputs,
        accumulate_grad,
        /*min_topo_nr=*/0);
  }

  // 3. Compute dependencies
  compute_dependencies(graph_root.get(), *graph_task, /*min_topo_nr=*/0);

  // 4. Push root task to ready queue
  auto root_task = NodeTask(
      graph_task,
      graph_root,
      InputBuffer(graph_root->num_inputs()));

  // Populate root task with initial gradients
  for (size_t i = 0; i < inputs.size(); i++) {
    root_task.inputs_.buffer[i] = inputs[i];
  }

  cpu_ready_queue->push(std::move(root_task));

  // 5. Execute backward on current thread
  graph_task->owner_ = worker_device;
  ++total_depth;
  ++current_depth;

  try {
    thread_main(graph_task);
  } catch (...) {
    --current_depth;
    --total_depth;
    throw;
  }

  --current_depth;
  --total_depth;

  // 6. Return captured gradients (if outputs specified)
  return std::move(graph_task->captured_vars_);
}
```

### 2. compute_dependencies()

Traverse graph to count how many inputs each node needs:

```cpp
void Engine::compute_dependencies(
    Node* root,
    GraphTask& task,
    uint64_t min_topo_nr) {

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

      // Pruning: skip nodes below minimum topological number
      if (next_fn->topological_nr() < min_topo_nr) continue;

      // Increment dependency count
      task.dependencies_[next_fn]++;

      // Add to graph
      task.nodes_in_graph_.insert(next_fn);

      // Continue traversal
      if (seen.insert(next_fn).second) {
        queue.push(next_fn);
      }
    }
  }
}
```

**Result**:
- `task.dependencies_[node]` = number of inputs this node needs before execution
- `task.nodes_in_graph_` = set of all reachable nodes

### 3. thread_main()

Main worker loop that processes tasks from the ready queue:

```cpp
void Engine::thread_main(const std::shared_ptr<GraphTask>& graph_task) {
  // For device threads: graph_task is nullptr (runs forever)
  // For user thread: graph_task is the current backward (runs until complete)

  while (graph_task == nullptr || !graph_task->future_result_->completed()) {
    // Pop task from ready queue (blocks if empty)
    NodeTask task = local_ready_queue->pop();

    // Check for shutdown
    if (task.isShutdownTask_) {
      break;
    }

    std::shared_ptr<GraphTask> local_graph_task = task.base_.lock();
    if (!local_graph_task) {
      // Graph task no longer valid, skip
      continue;
    }

    // Set device for this thread
    set_device(worker_device);

    if (task.fn_ && !local_graph_task->has_error_.load()) {
      // Set thread-local state (grad_mode, etc.)
      at::ThreadLocalStateGuard tls_guard(local_graph_task->thread_locals_);

      try {
        // Evaluate the function
        GraphTaskGuard guard(local_graph_task);
        NodeGuard ndguard(task.fn_);

        evaluate_function(
            local_graph_task,
            task.fn_.get(),
            task.inputs_,
            local_graph_task->cpu_ready_queue_);
      } catch (std::exception& e) {
        thread_on_exception(local_graph_task, task.fn_, e);
      }
    }

    // Decrement outstanding tasks
    --local_graph_task->outstanding_tasks_;

    // Check if graph task is complete
    if (local_graph_task->completed()) {
      local_graph_task->mark_as_completed_and_run_post_processing();

      // Wake up owner thread if it's waiting
      if (graph_task != local_graph_task) {
        // Push dummy task to owner's queue to wake it up
        auto owner = local_graph_task->owner_;
        ready_queue_by_index(cpu_ready_queue, owner)->push(
            NodeTask(local_graph_task, nullptr, InputBuffer(0)));
      }
    }
  }
}
```

### 4. evaluate_function()

Executes a single gradient function and distributes outputs:

```cpp
void Engine::evaluate_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputs,
    const std::shared_ptr<ReadyQueue>& cpu_ready_queue) {

  // 1. Stream synchronization for GPU tensors
  auto opt_parent_stream = (*func).stream();
  c10::OptionalStreamGuard parent_stream_guard{opt_parent_stream};

  for (size_t pos = 0; pos < inputs.ready_events.size(); ++pos) {
    if (inputs.ready_streams[pos] &&
        *opt_parent_stream != *inputs.ready_streams[pos]) {
      opt_parent_stream->wait(inputs.ready_events[pos].value());
    }
  }

  // 2. Handle exec_info (selective backward)
  auto& exec_info_ = graph_task->exec_info_;
  if (!exec_info_.empty()) {
    auto& fn_info = exec_info_.at(func);

    // Capture gradients if requested
    if (auto* capture_vec = fn_info.captures_.get()) {
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      for (const auto& capture : *capture_vec) {
        graph_task->captured_vars_[capture.output_idx_] =
            inputs.buffer[capture.input_idx_];
      }
    }

    // Skip execution if not needed
    if (!fn_info.needed_) {
      return;
    }
  }

  // 3. Call the gradient function
  auto outputs = call_function(graph_task, func, inputs);

  // 4. Release saved variables if not keeping graph
  if (!graph_task->keep_graph_) {
    func->release_variables();
  }

  // 5. No outputs = leaf node (AccumulateGrad), we're done
  if (outputs.size() == 0) {
    if (opt_parent_stream) {
      std::lock_guard<std::mutex> lock(graph_task->mutex_);
      graph_task->leaf_streams.emplace(*opt_parent_stream);
    }
    return;
  }

  // 6. Anomaly detection (check for NaN)
  if (AnomalyMode::is_enabled() && AnomalyMode::should_check_nan()) {
    for (const auto& output : outputs) {
      if (output.defined() && at::isnan(output).any().item<bool>()) {
        // Throw error with stack trace
      }
    }
  }

  // 7. Distribute outputs to next nodes
  std::lock_guard<std::mutex> lock(graph_task->mutex_);

  for (size_t i = 0; i < func->num_outputs(); i++) {
    const Edge& edge = func->next_edge(i);

    if (!edge.is_valid()) continue;
    if (!outputs[i].defined()) continue;

    Node* next_fn = edge.function.get();
    uint32_t input_nr = edge.input_nr;

    // Check if this node should execute
    if (!exec_info_.empty()) {
      auto it = exec_info_.find(next_fn);
      if (it == exec_info_.end() || !it->second.should_execute()) {
        continue;
      }
    }

    // Get or create InputBuffer for next node
    auto& input_buffer = graph_task->not_ready_[next_fn];
    if (input_buffer.buffer.empty()) {
      input_buffer = InputBuffer(next_fn->num_inputs());
    }

    // Add gradient to input buffer
    input_buffer.add(
        input_nr,
        std::move(outputs[i]),
        opt_parent_stream,
        next_fn->stream(),
        next_fn);

    // Decrement dependency count
    auto& dependencies = graph_task->dependencies_;
    auto dep_it = dependencies.find(next_fn);

    if (--dep_it->second == 0) {
      // All dependencies satisfied, ready to execute
      dependencies.erase(dep_it);

      // Create NodeTask and push to appropriate queue
      auto next_task = NodeTask(
          graph_task,
          std::shared_ptr<Node>(next_fn->getptr()),
          std::move(input_buffer));

      graph_task->not_ready_.erase(next_fn);

      // Choose queue based on device
      auto device = next_fn->device();
      auto queue = ready_queue(cpu_ready_queue, device);
      queue->push(std::move(next_task));
    }
  }
}
```

### 5. call_function()

Executes the node's backward function with hooks:

```cpp
static variable_list call_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputBuffer) {

  auto& fn = *func;

  // 1. Call tensor pre-hooks
  auto inputs = call_tensor_pre_hooks(
      fn, InputBuffer::variables(std::move(inputBuffer)));

  // 2. Call pre-hooks
  inputs = call_pre_hooks(fn, std::move(inputs));

  // 3. Signal that variables will be released
  if (!graph_task->keep_graph_) {
    fn.will_release_variables();
  }

  // 4. Execute the function
  const auto has_post_hooks = !fn.post_hooks().empty();
  variable_list outputs;

  if (has_post_hooks) {
    auto inputs_copy = inputs;
    outputs = fn(std::move(inputs_copy));
  } else {
    outputs = fn(std::move(inputs));
  }

  // 5. Validate outputs (shape, dtype, device)
  validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) {
    std::ostringstream ss;
    ss << "Function " << fn.name() << " returned an " << msg;
    return ss.str();
  });

  // 6. Call post-hooks
  return call_post_hooks(fn, std::move(outputs), inputs, has_post_hooks);
}
```

## Thread Model

### Device Threads

**File**: [torch/csrc/autograd/engine.cpp](../reference/pytorch/torch/csrc/autograd/engine.cpp#L98-L176)

PyTorch creates dedicated threads for each device:

```cpp
// Thread-local variable tracking which device this thread handles
static thread_local int worker_device = NO_DEVICE;

void Engine::start_device_threads() {
  // Create one thread per accelerator device (CUDA, XLA, etc.)
  for (size_t i = 0; i < device_ready_queues_.size(); ++i) {
    std::thread t([this, i] {
      thread_init(i, device_ready_queues_[i], /*should_increment=*/true);
    });
    t.detach();  // Long-running daemon threads
  }
}

void Engine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {

  worker_device = device;  // Thread-local assignment
  init_local_ready_queue(ready_queue);

  std::shared_ptr<GraphTask> graph_task = nullptr;
  thread_main(graph_task);  // Run forever (graph_task == nullptr)
}
```

**Device Assignment**:
- `worker_device = NO_DEVICE` (-2): Uninitialized
- `worker_device = CPU_DEVICE` (-1): CPU device
- `worker_device >= 0`: Device index (e.g., CUDA device 0)

**Note**: CPU work is NOT handled by a dedicated thread. Instead, the user's calling thread executes CPU work directly (synchronous execution).

### Reentrant Backwards

**File**: [torch/csrc/autograd/engine.cpp](../reference/pytorch/torch/csrc/autograd/engine.cpp#L147-L176)

Calling `backward()` inside a backward hook creates reentrant execution:

```python
def hook(grad):
    # Calling backward inside the hook
    some_other_loss.backward()
    return grad

x.register_hook(hook)
```

**Problem**:
- Worker thread executing the hook needs to block until nested backward completes
- But worker thread is also responsible for executing tasks in the nested backward
- Deadlock if thread blocks waiting for itself!

**Solution**:
1. Maintain a **thread pool** of idle workers
2. When reentrant backward detected, current thread blocks
3. Wake up worker from pool to handle tasks
4. New worker executes nested backward using **same ReadyQueue**
5. When nested backward completes, pool worker goes idle, original thread resumes

**Depth Limit**:
```cpp
static constexpr int MAX_DEPTH = 60;

if (task->reentrant_depth_ >= MAX_DEPTH) {
  // Spawn new thread to avoid stack overflow
  std::thread([task]() {
    execute_graph_task(task);
  }).detach();
}
```

**Thread-Local State**:
```cpp
static thread_local int current_depth = 0;  // Nesting level in this thread
static thread_local int total_depth = 0;    // Total nesting across all threads
```

### Stream Synchronization

**File**: [torch/csrc/autograd/engine.cpp](../reference/pytorch/torch/csrc/autograd/engine.cpp#L177-L207)

For GPU execution, backward operations run on the **same streams as forward**:

```
Forward Pass:
  Stream 0: x = input.cuda()
  Stream 0: y = layer1(x)
  Stream 1: z = layer2(y)  # Assume different stream

Backward Pass:
  Stream 1: grad_z computed
  Stream 0: grad_y computed
  # Must sync: Stream 1 must wait for Stream 0 before using grad_y
```

**Mechanism**:

1. **Forward**: Record which stream each operation ran on
2. **Backward**: Each `Node` has a `stream()` method returning the forward stream
3. **Synchronization**: Before calling `node->apply()`, wait for all input gradients to be ready on their respective streams

```cpp
// In evaluate_function()
auto opt_parent_stream = (*func).stream();  // Stream this node ran on in forward

for (size_t pos = 0; pos < inputs.ready_events.size(); ++pos) {
  auto& opt_ready_stream = inputs.ready_streams[pos];
  auto& opt_ready_event = inputs.ready_events[pos];

  if (*opt_parent_stream != *opt_ready_stream) {
    // Different streams, need to sync
    opt_parent_stream->wait(opt_ready_event.value());
  }
}
```

**Leaf Streams**:
At the end of backward, synchronize user's current streams with leaf streams:

```cpp
void GraphTask::exec_post_processing() {
  // For each leaf stream (stream that executed a leaf node)
  for (const c10::Stream& leaf_stream : leaf_streams) {
    // Sync caller's current stream (on same device) with leaf stream
    auto caller_stream = caller_current_streams_[leaf_stream.device_index()];
    if (caller_stream.has_value()) {
      caller_stream.value().wait(leaf_stream);
    }
  }
}
```

This ensures that after `loss.backward()` returns, all gradient computations are complete on the user's streams.

## Execution Example

**Forward Pass**:
```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = x * y        # MulBackward0, sequence_nr = 100
w = z + 1        # AddBackward0, sequence_nr = 101
loss = w.sum()   # SumBackward0, sequence_nr = 102
```

**Backward Execution**:

```
Step 1: Engine::execute() called
  - Create GraphTask
  - compute_dependencies():
      SumBackward0: dependencies = 0
      AddBackward0: dependencies = 1
      MulBackward0: dependencies = 1
      AccumulateGrad(x): dependencies = 1
      AccumulateGrad(y): dependencies = 1
  - Push SumBackward0 to ready queue with grad = 1.0

Step 2: thread_main() pops SumBackward0
  - evaluate_function(SumBackward0):
      - call_function(): outputs = [1.0]
      - Distribute to next nodes:
          AddBackward0.not_ready_[0] = 1.0
          dependencies[AddBackward0]-- → 0 (ready!)
      - Push AddBackward0 to ready queue

Step 3: thread_main() pops AddBackward0
  - evaluate_function(AddBackward0):
      - call_function(): outputs = [1.0]
      - Distribute to next nodes:
          MulBackward0.not_ready_[0] = 1.0
          dependencies[MulBackward0]-- → 0 (ready!)
      - Push MulBackward0 to ready queue

Step 4: thread_main() pops MulBackward0
  - evaluate_function(MulBackward0):
      - call_function(): outputs = [y, x] = [3.0, 2.0]
      - Distribute to next nodes:
          AccumulateGrad(x).not_ready_[0] = 3.0
          dependencies[AccumulateGrad(x)]-- → 0 (ready!)
          AccumulateGrad(y).not_ready_[0] = 2.0
          dependencies[AccumulateGrad(y)]-- → 0 (ready!)
      - Push both AccumulateGrad nodes to ready queue

Step 5: thread_main() pops AccumulateGrad(x) (sequence_nr = UINT64_MAX, highest priority)
  - evaluate_function(AccumulateGrad(x)):
      - call_function(): x.grad = 3.0, outputs = []
      - No next edges (leaf node)
      - Record leaf_stream

Step 6: thread_main() pops AccumulateGrad(y)
  - evaluate_function(AccumulateGrad(y)):
      - call_function(): y.grad = 2.0, outputs = []
      - No next edges (leaf node)
      - Record leaf_stream

Step 7: All nodes complete
  - outstanding_tasks = 0
  - GraphTask::mark_as_completed_and_run_post_processing()
      - Sync leaf_streams with caller streams
      - Set future_result_
      - Run final_callbacks_
  - thread_main() exits

Result: x.grad = 3.0, y.grad = 2.0
```

## Selective Backward

When `torch.autograd.grad()` is called with `inputs=` parameter:

```python
x = torch.randn(5, requires_grad=True)
y = torch.randn(5, requires_grad=True)
z = torch.randn(5, requires_grad=True)

out = x * y + z
grads = torch.autograd.grad(out.sum(), [x])  # Only compute grad for x
```

**Execution**:

1. **Mark needed nodes**: Traverse backward from `x` to mark which nodes contribute to `x`'s gradient
2. **Set exec_info**: `exec_info_[node].needed_ = true` for marked nodes
3. **During backward**: Skip nodes where `exec_info_[node].should_execute() == false`

**Code**:
```cpp
void GraphTask::init_to_execute(
    Node& graph_root,
    const edge_list& outputs,
    bool accumulate_grad,
    uint64_t min_topo_nr) {

  // Mark outputs as needed
  for (const auto& edge : outputs) {
    exec_info_[edge.function.get()].needed_ = true;
  }

  // Backward traverse from outputs to roots
  std::queue<Node*> queue;
  for (const auto& edge : outputs) {
    queue.push(edge.function.get());
  }

  while (!queue.empty()) {
    Node* fn = queue.front();
    queue.pop();

    for (const auto& next_edge : fn->next_edges()) {
      if (!next_edge.is_valid()) continue;

      Node* next_fn = next_edge.function.get();

      // Mark as needed
      auto& info = exec_info_[next_fn];
      if (!info.needed_) {
        info.needed_ = true;
        queue.push(next_fn);
      }
    }
  }
}
```

## Error Handling

**Exception Propagation**:

```cpp
void Engine::thread_on_exception(
    const std::shared_ptr<GraphTask>& graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {

  // Mark error
  graph_task->set_exception(std::current_exception(), fn);

  // If exit_on_error, stop all workers immediately
  if (graph_task->exit_on_error_) {
    graph_task->has_error_.store(true);
    graph_task->future_result_->setError(std::current_exception());
  }
}
```

**Behavior**:
- By default, first exception is saved, other workers continue
- When all workers finish, exception is re-thrown to user
- With `exit_on_error=True`, all workers stop immediately

## MLX Porting Considerations

### MLX Execution Model

MLX uses a **functional** approach with lazy evaluation:

```python
import mlx.core as mx

# MLX: Define gradient function
def loss_fn(params):
    return compute_loss(params)

grad_fn = mx.grad(loss_fn)

# Compute gradients (lazy, builds computational graph)
grads = grad_fn(params)

# Force evaluation
mx.eval(grads)
```

### Key Differences

| Aspect | PyTorch | MLX |
|--------|---------|-----|
| **Execution** | Eager with explicit graph | Lazy with implicit graph |
| **Scheduling** | Priority queue + dependencies | Automatic via lazy evaluation |
| **Threading** | Device threads + user thread | Automatic parallelization |
| **Stream Sync** | Explicit with events | Automatic via dependency analysis |
| **Reentrant** | Complex thread pool | Not applicable (functional) |
| **Selective Backward** | exec_info filtering | grad(argnums=[...]) |

### Porting Challenges

**1. Eager vs Lazy**:

PyTorch builds graph during forward, MLX builds graph during grad():

```python
# PyTorch: Graph exists after forward
y = model(x)  # Graph built here
y.backward()  # Traverse existing graph

# MLX: Graph built when grad() called
def forward(params, x):
    return model(params, x)

grad_fn = mx.grad(forward)  # Define gradient function
grads = grad_fn(params, x)  # Graph built and executed here
```

**2. Multi-Device**:

PyTorch explicitly manages devices; MLX automatically handles Metal:

```python
# PyTorch: Explicit device threads
# CPU work on user thread
# CUDA work on CUDA threads
# Sync with events

# MLX: Automatic
# All work on Metal
# Automatic dependency tracking
```

**3. No Explicit Engine**:

MLX doesn't expose an "Engine" class. The compilation infrastructure handles scheduling internally.

### MLX Implementation Sketch

For compatibility, create a minimal engine:

```python
class MLXEngine:
    """Minimal engine for PyTorch compatibility"""

    def execute(self, roots, inputs, keep_graph=False, create_graph=False):
        # roots are loss tensors
        # inputs are initial gradients

        # Build gradient function
        def backward_fn():
            # Compute gradients using mx.grad()
            return mx.grad(self._forward_fn)(*args)

        # Execute
        grads = backward_fn()

        # Accumulate into .grad attributes (PyTorch compatibility)
        for param, grad in zip(params, grads):
            if param.grad is None:
                param.grad = grad
            else:
                param.grad = param.grad + grad

        return grads
```

**Recommendation**: Don't emulate PyTorch's engine. Instead:
- Use `mx.grad()` for gradient computation
- Use `mx.value_and_grad()` for forward + backward in one call
- Leverage MLX's automatic parallelization
- Let users adopt functional style for new code
- Provide compatibility layer only for model loading/conversion

## Critical File References

- [torch/csrc/autograd/engine.h](../reference/pytorch/torch/csrc/autograd/engine.h): Engine class, ReadyQueue, NodeTask
- [torch/csrc/autograd/engine.cpp](../reference/pytorch/torch/csrc/autograd/engine.cpp): Full implementation (1777 lines)
- [torch/csrc/autograd/graph_task.h](../reference/pytorch/torch/csrc/autograd/graph_task.h): GraphTask, ExecInfo, dependency tracking
- [torch/csrc/autograd/input_buffer.h](../reference/pytorch/torch/csrc/autograd/input_buffer.h): InputBuffer for gradient accumulation
- [torch/csrc/autograd/function.h](../reference/pytorch/torch/csrc/autograd/function.h): Node base class, hooks
- [torch/csrc/autograd/functions/accumulate_grad.h](../reference/pytorch/torch/csrc/autograd/functions/accumulate_grad.h): AccumulateGrad implementation

## Next Steps

This completes the backward engine documentation. The remaining autograd documents will cover:

1. **gradient-formulas.md**: How derivatives are defined in derivatives.yaml, common derivative patterns
2. **custom-functions.md**: torch.autograd.Function API for implementing custom backward passes
