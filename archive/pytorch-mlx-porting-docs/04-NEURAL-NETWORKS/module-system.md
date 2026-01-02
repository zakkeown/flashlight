# PyTorch nn.Module System

## Purpose

This document provides a comprehensive analysis of PyTorch's `nn.Module` class, the foundation of all neural network layers and models in PyTorch. Understanding `nn.Module` is critical for MLX porting because:

1. It defines the standard pattern for model composition
2. It establishes the parameter management system
3. It provides the training/evaluation mode infrastructure
4. It serves as the template for translating PyTorch models to MLX

## Architecture Overview

### nn.Module: The Base Class

`nn.Module` is Python's foundational abstraction for neural network components. Every layer, model, and container in PyTorch inherits from it.

**Source**: [torch/nn/modules/module.py](../reference/pytorch/torch/nn/modules/module.py)

```
┌─────────────────────────────────────────────────────────────┐
│                      nn.Module                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Internal State (OrderedDict)                        │  │
│  │  • _parameters: trainable weights                    │  │
│  │  • _buffers: non-trainable state (running stats)     │  │
│  │  • _modules: child modules (hierarchical)            │  │
│  │  • _forward_hooks: callbacks after forward           │  │
│  │  • _backward_hooks: callbacks after backward         │  │
│  │  • training: bool (train vs eval mode)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Core Methods                                        │  │
│  │  • __init__(): Initialize internal dicts             │  │
│  │  • forward(*args): Define computation (override!)    │  │
│  │  • __call__(*args): Wrapper around forward + hooks   │  │
│  │  • parameters(): Iterator over trainable params      │  │
│  │  • state_dict(): Serialization of all state          │  │
│  │  • load_state_dict(): Deserialization               │  │
│  │  • to(device): Move all tensors to device            │  │
│  │  • train()/eval(): Switch modes                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↑
                          │ (inherits from)
           ┌──────────────┼──────────────┐
           │              │              │
      ┌────────┐    ┌───────────┐  ┌──────────┐
      │ Linear │    │   Conv2d  │  │  ReLU    │
      └────────┘    └───────────┘  └──────────┘
           │              │              │
           └──────────────┼──────────────┘
                          │ (composed into)
                   ┌──────────────┐
                   │ MyModel      │
                   │ (nn.Module)  │
                   └──────────────┘
```

**Design Philosophy**:
- **Hierarchical Composition**: Modules contain other modules
- **Automatic Parameter Discovery**: Parameters are auto-registered
- **Hook System**: Extensible callback system
- **Device Transparency**: Single `.to(device)` moves entire model

## Core Components

### Component 1: Parameter Management

#### nn.Parameter Class

**Source**: [torch/nn/parameter.py](../reference/pytorch/torch/nn/parameter.py)

```python
class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag
        t = data.detach().requires_grad_(requires_grad)
        t._is_param = True
        return t
```

**Key Features**:
1. **Tensor Subclass**: `Parameter` is a `Tensor` with special metadata
2. **Auto-registration**: Assigned parameters are automatically tracked
3. **Gradient Tracking**: `requires_grad=True` by default
4. **Serialization**: Included in `state_dict()` automatically

**Usage Pattern**:
```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Create parameters (auto-registered)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias
```

#### register_parameter() Method

**Source**: [torch/nn/modules/module.py](../reference/pytorch/torch/nn/modules/module.py:590-640)

```python
def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
    r"""Add a parameter to the module.

    Args:
        name (str): name of the parameter. The parameter can be accessed
            from this module using the given name
        param (Parameter or None): parameter to be added to the module.
    """
    if "_parameters" not in self.__dict__:
        raise AttributeError(
            "cannot assign parameter before Module.__init__() call"
        )

    elif not isinstance(name, str):
        raise TypeError(
            f"parameter name should be a string. Got {torch.typename(name)}"
        )
    elif "." in name:
        raise KeyError('parameter name can\'t contain "."')
    elif name == "":
        raise KeyError('parameter name can\'t be empty string ""')
    elif hasattr(self, name) and name not in self._parameters:
        raise KeyError(f"attribute '{name}' already exists")

    if param is None:
        self._parameters[name] = None
    elif not isinstance(param, Parameter):
        raise TypeError(
            f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
            "(torch.nn.Parameter or None required)"
        )
    elif param.grad_fn:
        raise ValueError(
            f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
            f"parameters must be created explicitly."
        )
    else:
        # Call global registration hooks
        for hook in _global_parameter_registration_hooks.values():
            output = hook(self, name, param)
            if output is not None:
                param = output
        self._parameters[name] = param
```

**Design Patterns**:
1. **Validation**: Strict checks on parameter names and types
2. **Leaf Tensor Requirement**: Parameters must be leaf tensors (no grad_fn)
3. **Hook Support**: Global hooks can intercept parameter registration
4. **Storage**: Parameters stored in `OrderedDict` for ordered iteration

#### parameters() Iterator

```python
def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    r"""Return an iterator over module parameters.

    This is typically passed to an optimizer.

    Args:
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        Parameter: module parameter
    """
    for name, param in self.named_parameters(recurse=recurse):
        yield param

def named_parameters(self, prefix: str = '', recurse: bool = True,
                      remove_duplicate: bool = True) -> Iterator[tuple[str, Parameter]]:
    r"""Return an iterator over module parameters, yielding both the
    name of the parameter as well as the parameter itself.

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules.
        remove_duplicate (bool): whether to remove duplicated parameters
    """
    gen = self._named_members(
        lambda module: module._parameters.items(),
        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    )
    yield from gen
```

**Usage**:
```python
model = MyModel()

# Get all parameters
all_params = list(model.parameters())

# Pass to optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Named parameters (with hierarchical names)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
# Output:
# encoder.weight: torch.Size([512, 256])
# encoder.bias: torch.Size([512])
# decoder.weight: torch.Size([256, 512])
# decoder.bias: torch.Size([256])
```

### Component 2: Buffer Management

#### Buffers vs Parameters

**Buffers** are non-trainable tensors that are part of module state (e.g., BatchNorm running statistics).

```python
class Buffer(torch.Tensor, metaclass=_BufferMeta):
    r"""A kind of Tensor that should not be considered a model parameter.

    Buffers are :class:`~torch.Tensor` subclasses that have a very special
    property when used with :class:`Module` s -- when they're assigned as
    Module attributes they are automatically added to the list of its buffers,
    and will appear e.g. in :meth:`~torch.nn.Module.buffers` iterator.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = torch.empty(0)

        t = data.detach().requires_grad_(data.requires_grad)
        t.persistent = persistent
        t._is_buffer = True
        return t
```

#### register_buffer() Method

```python
def register_buffer(self, name: str, tensor: Optional[Tensor],
                    persistent: bool = True) -> None:
    r"""Add a buffer to the module.

    This is typically used to register a buffer that should not be
    considered a model parameter. For example, BatchNorm's ``running_mean``
    is not a parameter, but is part of the module's state.

    Buffers can be accessed as attributes using the given name.

    Args:
        name (str): name of the buffer.
        tensor (Tensor or None): buffer to be registered. If ``None``, then
            operations that run on buffers, such as :attr:`cuda`, are ignored.
        persistent (bool): whether the buffer is part of this module's
            :attr:`state_dict`. Default: ``True``
    """
    if "_buffers" not in self.__dict__:
        raise AttributeError(
            "cannot assign buffer before Module.__init__() call"
        )
    elif not isinstance(name, str):
        raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
    elif "." in name:
        raise KeyError('buffer name can\'t contain "."')
    elif name == "":
        raise KeyError('buffer name can\'t be empty string ""')
    elif hasattr(self, name) and name not in self._buffers:
        raise KeyError(f"attribute '{name}' already exists")

    if tensor is not None and not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
            "(torch.Tensor or None required)"
        )
    else:
        for hook in _global_buffer_registration_hooks.values():
            output = hook(self, name, tensor)
            if output is not None:
                tensor = output
        self._buffers[name] = tensor
```

**Example: BatchNorm Buffers**:
```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Parameters (trainable)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Buffers (non-trainable, but part of state)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # Update running statistics (buffers)
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
            self.num_batches_tracked += 1
        else:
            # Use running statistics in eval mode
            mean = self.running_mean
            var = self.running_var

        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias
```

**Comparison: Parameters vs Buffers**:

| Aspect | Parameters | Buffers |
|--------|-----------|---------|
| **Trainable** | Yes (requires_grad=True) | No (requires_grad=False) |
| **In state_dict** | Always | Only if persistent=True |
| **Updated by optimizer** | Yes | No (manual updates only) |
| **Examples** | Weights, biases | Running mean/var, anchor boxes |

### Component 3: Submodule Management

#### add_module() Method

**Source**: [torch/nn/modules/module.py](../reference/pytorch/torch/nn/modules/module.py:642-668)

```python
def add_module(self, name: str, module: Optional["Module"]) -> None:
    r"""Add a child module to the current module.

    The module can be accessed as an attribute using the given name.

    Args:
        name (str): name of the child module.
        module (Module): child module to be added to the module.
    """
    if not isinstance(module, Module) and module is not None:
        raise TypeError(f"{torch.typename(module)} is not a Module subclass")
    elif not isinstance(name, str):
        raise TypeError(f"module name should be a string. Got {torch.typename(name)}")
    elif hasattr(self, name) and name not in self._modules:
        raise KeyError(f"attribute '{name}' already exists")
    elif "." in name:
        raise KeyError(f'module name can\'t contain ".", got: {name}')
    elif name == "":
        raise KeyError('module name can\'t be empty string ""')

    for hook in _global_module_registration_hooks.values():
        output = hook(self, name, module)
        if output is not None:
            module = output
    self._modules[name] = module
```

#### Hierarchical Module Composition

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Submodules auto-registered when assigned
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.decoder = nn.Linear(256, 10)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# Module hierarchy:
# MyModel/
#   encoder/
#     0/ (Linear)
#       weight (Parameter)
#       bias (Parameter)
#     1/ (ReLU)
#     2/ (Linear)
#       weight (Parameter)
#       bias (Parameter)
#   decoder/
#     weight (Parameter)
#     bias (Parameter)
```

#### modules() Iterator

```python
def modules(self) -> Iterator['Module']:
    r"""Return an iterator over all modules in the network.

    Yields:
        Module: a module in the network

    Note:
        Duplicate modules are returned only once. In the following
        example, ``l`` will be returned only once.
    """
    for _, module in self.named_modules():
        yield module

def named_modules(self, memo: Optional[set[Self]] = None, prefix: str = '',
                   remove_duplicate: bool = True):
    r"""Return an iterator over all modules in the network, yielding
    both the name of the module as well as the module itself.

    Yields:
        (str, Module): Tuple of name and module
    """
    if memo is None:
        memo = set()
    if self not in memo:
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(memo, submodule_prefix, remove_duplicate)
```

**Usage**:
```python
model = MyModel()

# Iterate over all modules
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")
# Output:
# : MyModel
# encoder: Sequential
# encoder.0: Linear
# encoder.1: ReLU
# encoder.2: Linear
# decoder: Linear
```

### Component 4: Forward Pass and Hooks

#### __call__() vs forward()

Users **override `forward()`**, but **call the module** (which invokes `__call__()`):

```python
class Module:
    def __call__(self, *args, **kwargs):
        # Pre-forward hooks
        for hook_id, hook in self._forward_pre_hooks.items():
            result = hook(self, args)
            if result is not None:
                args = result

        # Actual forward pass
        result = self.forward(*args, **kwargs)

        # Post-forward hooks
        for hook_id, hook in self._forward_hooks.items():
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result

        # Backward hooks registration
        if self._backward_hooks or self._backward_pre_hooks:
            # Register backward hooks on the result tensor
            result.grad_fn.register_hook(...)

        return result

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclass must override forward()")
```

**Key Pattern**: Users implement `forward()`, but invoke the module as a callable:

```python
class MyModule(nn.Module):
    def forward(self, x):  # User defines this
        return x * 2

module = MyModule()
result = module(input)  # User calls this (invokes __call__)
```

#### Forward Hooks

```python
def register_forward_pre_hook(self, hook: Callable[..., None],
                               *, prepend: bool = False,
                               with_kwargs: bool = False) -> RemovableHandle:
    r"""Register a forward pre-hook on the module.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, args) -> None or modified args
    """
    handle = RemovableHandle(self._forward_pre_hooks, ...)
    self._forward_pre_hooks[handle.id] = hook
    return handle

def register_forward_hook(self, hook: Callable[..., None],
                           *, prepend: bool = False,
                           with_kwargs: bool = False,
                           always_call: bool = False) -> RemovableHandle:
    r"""Register a forward hook on the module.

    The hook will be called every time after :func:`forward` has computed
    an output. It should have the following signature::

        hook(module, args, output) -> None or modified output
    """
    handle = RemovableHandle(self._forward_hooks, ...)
    self._forward_hooks[handle.id] = hook
    return handle
```

**Use Cases**:
1. **Feature Extraction**: Extract intermediate activations
2. **Debugging**: Inspect inputs/outputs at each layer
3. **Visualization**: Capture activations for visualization
4. **Monitoring**: Track statistics during training

**Example: Feature Extraction**:
```python
model = torchvision.models.resnet50(pretrained=True)
features = {}

def get_features(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

# Register hooks on specific layers
model.layer1.register_forward_hook(get_features('layer1'))
model.layer2.register_forward_hook(get_features('layer2'))
model.layer3.register_forward_hook(get_features('layer3'))

# Forward pass
output = model(input_tensor)

# Now features dict contains intermediate activations
print(features['layer1'].shape)  # torch.Size([batch, 256, 56, 56])
print(features['layer2'].shape)  # torch.Size([batch, 512, 28, 28])
```

### Component 5: Training vs Evaluation Mode

#### train() and eval() Methods

```python
def train(self: T, mode: bool = True) -> T:
    r"""Set the module in training mode.

    This has any effect only on certain modules. See documentations of
    particular modules for details of their behaviors in training/evaluation
    mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    etc.

    Args:
        mode (bool): whether to set training mode (``True``) or evaluation
                     mode (``False``). Default: ``True``.

    Returns:
        Module: self
    """
    if not isinstance(mode, bool):
        raise ValueError("training mode is expected to be boolean")
    self.training = mode
    for module in self.children():
        module.train(mode)
    return self

def eval(self: T) -> T:
    r"""Set the module in evaluation mode.

    This has any effect only on certain modules. See documentations of
    particular modules for details of their behaviors in training/evaluation
    mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    etc.

    This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

    See :ref:`locally-disable-gradient-computation` for a comparison between
    `.eval()` and several similar mechanisms that may be confused with it.

    Returns:
        Module: self
    """
    return self.train(False)
```

**Key Behaviors**:
1. **Recursive Application**: Sets mode on all submodules
2. **Returns self**: Allows method chaining
3. **Persistent State**: `self.training` flag accessible in `forward()`

#### Mode-Dependent Modules

**Dropout**:
```python
class Dropout(nn.Module):
    def forward(self, x):
        if self.training:
            # Apply dropout mask during training
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            return x * mask / (1 - self.p)
        else:
            # No dropout during evaluation
            return x
```

**BatchNorm**:
```python
class BatchNorm(nn.Module):
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3], unbiased=False)
            # Update running statistics (buffers)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        return (x - mean) / torch.sqrt(var + eps) * self.weight + self.bias
```

**Usage Pattern**:
```python
model = MyModel()

# Training loop
model.train()  # Enable dropout, batch statistics computation
for batch in train_loader:
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()  # Disable dropout, use running statistics
with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
        accuracy = compute_accuracy(output, target)
```

### Component 6: Device Movement

#### to() Method

```python
def to(self, *args, **kwargs) -> Self:
    r"""Move and/or cast the parameters and buffers.

    This can be called as

    .. function:: to(device=None, dtype=None, non_blocking=False)
       :noindex:

    .. function:: to(dtype, non_blocking=False)
       :noindex:

    .. function:: to(tensor, non_blocking=False)
       :noindex:

    .. function:: to(memory_format=torch.channels_last)
       :noindex:

    Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
    floating point or complex :attr:`dtype`\\ s. In addition, this method will
    only cast the floating point or complex parameters and buffers to :attr:`dtype`
    (if given). The integral parameters and buffers will be moved
    :attr:`device`, if that is given, but with dtypes unchanged.

    .. note::
        This method modifies the module in-place.

    Args:
        device (:class:`torch.device`): the desired device of the parameters
            and buffers in this module
        dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
            the parameters and buffers in this module
        tensor (torch.Tensor): Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module
        memory_format (:class:`torch.memory_format`): the desired memory
            format for 4D parameters and buffers in this module

    Returns:
        Module: self
    """
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    if dtype is not None:
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError('nn.Module.to only accepts floating point or complex '
                            f'dtypes, but got desired dtype={dtype}')

    def convert(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, dtype, non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype, non_blocking)

    return self._apply(convert)
```

**Usage Patterns**:
```python
model = MyModel()

# Move to GPU
model = model.to('cuda')
model = model.to(torch.device('cuda:0'))

# Move to CPU
model = model.to('cpu')

# Change dtype
model = model.to(torch.float16)  # Half precision

# Move to GPU and change dtype
model = model.to(device='cuda', dtype=torch.float16)

# Copy device from tensor
model = model.to(some_tensor.device)

# MPS (Metal) device
model = model.to('mps')
```

**Recursive Application**:
```python
def _apply(self, fn):
    for module in self.children():
        module._apply(fn)

    # Apply to parameters
    for key, param in self._parameters.items():
        if param is not None:
            with torch.no_grad():
                param_applied = fn(param)
            self._parameters[key] = nn.Parameter(param_applied, param.requires_grad)

    # Apply to buffers
    for key, buf in self._buffers.items():
        if buf is not None:
            self._buffers[key] = fn(buf)

    return self
```

### Component 7: State Dict (Serialization)

#### state_dict() Method

```python
def state_dict(self, *, destination=None, prefix='', keep_vars=False):
    r"""Return a dictionary containing references to the whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    Parameters and buffers set to ``None`` are not included.

    .. warning::
        Currently ``state_dict()`` also accepts positional arguments for
        ``destination``, ``prefix`` and ``keep_vars`` in order. However,
        this is being deprecated and keyword arguments will be enforced in
        future releases.

    Returns:
        dict:
            a dictionary containing a whole state of the module

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> module.state_dict().keys()
        ['bias', 'weight']
    """
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()

    # Save module metadata
    local_metadata = {}
    if hasattr(self, '_save_to_state_dict'):
        self._save_to_state_dict(destination, prefix, keep_vars)
    # Save submodules recursively
    for name, module in self._modules.items():
        if module is not None:
            module.state_dict(destination=destination, prefix=prefix + name + '.',
                             keep_vars=keep_vars)
    return destination

def _save_to_state_dict(self, destination, prefix, keep_vars):
    r"""Save module state to `destination` dictionary.

    This method is called on every submodule in `state_dict`.

    In rare cases, subclasses can achieve class-specific behavior by
    overriding this method with custom logic.

    Args:
        destination (dict): a dict where state will be stored
        prefix (str): the prefix for parameters and buffers used in this
            module
    """
    # Save parameters
    for name, param in self._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    # Save buffers
    for name, buf in self._buffers.items():
        if buf is not None and buf.persistent:
            destination[prefix + name] = buf if keep_vars else buf.detach()
    # Save extra state
    extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
    if hasattr(self, "get_extra_state"):
        destination[extra_state_key] = self.get_extra_state()
```

**Usage: Checkpointing**:
```python
# Save model
model = MyModel()
torch.save(model.state_dict(), 'checkpoint.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# State dict structure
state = model.state_dict()
# OrderedDict([
#     ('encoder.0.weight', tensor(...)),
#     ('encoder.0.bias', tensor(...)),
#     ('encoder.2.weight', tensor(...)),
#     ('encoder.2.bias', tensor(...)),
#     ('decoder.weight', tensor(...)),
#     ('decoder.bias', tensor(...)),
# ])
```

#### load_state_dict() Method

```python
def load_state_dict(self, state_dict: Mapping[str, Any],
                    strict: bool = True, assign: bool = False):
    r"""Copy parameters and buffers from :attr:`state_dict` into
    this module and its descendants.

    If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    .. warning::
        If :attr:`assign` is ``True`` the optimizer must be created after
        the call to :attr:`load_state_dict` unless
        :func:`~torch.optim.Optimizer.zero_grad` is called immediately.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
        assign (bool, optional): When ``False``, the properties of the tensors
            in the current module are preserved while when ``True``, the
            properties of the Tensors in the state dict are preserved. The only
            exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
            for which the value from the module is preserved.
            Default: ``False``

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys
    """
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    error_msgs: list[str] = []

    # Recursive loading
    def load(module, local_state_dict, prefix=''):
        local_metadata = {}
        module._load_from_state_dict(
            local_state_dict, prefix, local_metadata, True, missing_keys,
            unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {k: v for k, v in local_state_dict.items()
                                    if k.startswith(child_prefix)}
                load(child, child_state_dict, child_prefix)

    load(self, state_dict)
    del load

    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join(f'"{k}"' for k in missing_keys)))

    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           self.__class__.__name__, "\n\t".join(error_msgs)))

    return _IncompatibleKeys(missing_keys, unexpected_keys)
```

**Error Handling**:
```python
# Strict loading (default)
try:
    model.load_state_dict(state_dict)
except RuntimeError as e:
    print(f"Loading failed: {e}")

# Non-strict loading (allows mismatches)
incompatible = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {incompatible.missing_keys}")
print(f"Unexpected keys: {incompatible.unexpected_keys}")
```

## MLX Module Patterns

### mlx.nn.Module Comparison

MLX provides a similar `nn.Module` abstraction, but with key differences:

```python
import mlx.core as mx
import mlx.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # MLX uses freeze/unfreeze instead of requires_grad
        self.weight = mx.random.normal((out_features, in_features))
        self.bias = mx.zeros((out_features,))

    def __call__(self, x):
        # MLX uses __call__ directly (no separate forward)
        return x @ self.weight.T + self.bias

# Usage
model = Linear(784, 10)

# Parameters (auto-discovered via tree traversal)
params = nn.utils.tree_flatten(model.parameters())

# Freeze/unfreeze (instead of requires_grad)
nn.utils.freeze(model.weight)
nn.utils.unfreeze(model.weight)
```

**Key Differences**:

| Aspect | PyTorch nn.Module | MLX nn.Module |
|--------|-------------------|---------------|
| **Forward Method** | `forward()` (called via `__call__`) | `__call__()` directly |
| **Parameter Discovery** | `_parameters` OrderedDict | Tree traversal (pytree) |
| **Gradient Control** | `requires_grad` flag | `freeze()`/`unfreeze()` |
| **Training Mode** | `.train()` / `.eval()` | No built-in mode switching |
| **Device Movement** | `.to(device)` | Unified memory (no movement) |
| **State Dict** | `state_dict()` / `load_state_dict()` | `save_weights()` / `load_weights()` |
| **Hooks** | Extensive hook system | Minimal (functional transforms) |

### Migration Pattern: PyTorch → MLX

**PyTorch Pattern**:
```python
class PyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Usage
model = PyTorchModel().to('cuda')
model.train()
output = model(input)
```

**MLX Equivalent**:
```python
class MLXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        return self.fc2(x)

# Usage
model = MLXModel()  # No device movement needed
output = model(input)
```

**Parameter Access**:
```python
# PyTorch
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# MLX
params = model.parameters()
for path, param in nn.utils.tree_flatten_with_path(params):
    print(f"{'.'.join(path)}: {param.shape}")
```

**Checkpointing**:
```python
# PyTorch
torch.save(model.state_dict(), 'checkpoint.pth')
model.load_state_dict(torch.load('checkpoint.pth'))

# MLX
model.save_weights('checkpoint.npz')
model.load_weights('checkpoint.npz')
```

## Critical File References

### PyTorch Module System Core
- [torch/nn/modules/module.py](../reference/pytorch/torch/nn/modules/module.py) - nn.Module base class
- [torch/nn/parameter.py](../reference/pytorch/torch/nn/parameter.py) - Parameter and Buffer classes
- [torch/nn/modules/container.py](../reference/pytorch/torch/nn/modules/container.py) - Sequential, ModuleList, ModuleDict
- [torch/utils/hooks.py](../reference/pytorch/torch/utils/hooks.py) - Hook infrastructure

### Common Module Implementations
- [torch/nn/modules/linear.py](../reference/pytorch/torch/nn/modules/linear.py) - Linear layers
- [torch/nn/modules/conv.py](../reference/pytorch/torch/nn/modules/conv.py) - Convolution layers
- [torch/nn/modules/batchnorm.py](../reference/pytorch/torch/nn/modules/batchnorm.py) - BatchNorm layers
- [torch/nn/modules/dropout.py](../reference/pytorch/torch/nn/modules/dropout.py) - Dropout layers

## Next Steps

To continue understanding PyTorch → MLX porting:

1. **Read [04-NEURAL-NETWORKS/layers-reference/linear.md](layers-reference/linear.md)** - Concrete layer implementation
2. **Read [05-TRAINING/optimizer-base.md](../05-TRAINING/optimizer-base.md)** - How parameters interact with optimizers
3. **Read [08-PORTING-GUIDE/implementation-roadmap.md](../08-PORTING-GUIDE/implementation-roadmap.md)** - Step-by-step porting plan
4. **Study MLX nn.Module source** - See MLX's simpler approach

## Summary

PyTorch's `nn.Module` system provides a robust, feature-rich foundation for neural network construction. Key takeaways for MLX porting:

### PyTorch nn.Module Strengths
- ✅ Automatic parameter discovery and registration
- ✅ Hierarchical module composition with named access
- ✅ Extensive hook system for debugging and feature extraction
- ✅ Train/eval mode infrastructure for layers like Dropout and BatchNorm
- ✅ Comprehensive state dict serialization

### PyTorch nn.Module Complexity (MLX Simplifications)
- ❌ Separate `forward()` method (MLX uses `__call__()` directly)
- ❌ Complex hook system (MLX prefers functional transforms)
- ❌ Device movement API (unnecessary with unified memory)
- ❌ OrderedDict storage (MLX uses pytree traversal)

### Porting Strategy
1. **Core Pattern**: MLX's `nn.Module` mirrors PyTorch's hierarchical composition
2. **Simplifications**: Eliminate device management, simplify hook system
3. **Parameter Management**: Use MLX's freeze/unfreeze instead of requires_grad
4. **Mode Switching**: Implement train/eval as needed per-layer (no global flag)
5. **State Persistence**: Adapt state_dict concept to MLX's save_weights/load_weights

The module system is PyTorch's greatest strength - it provides a clean, composable way to build models. MLX inherits this pattern while simplifying aspects made unnecessary by Apple Silicon's unified memory architecture.
