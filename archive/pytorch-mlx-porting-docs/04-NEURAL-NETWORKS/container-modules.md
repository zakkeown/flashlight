# Container Modules

## Overview

Container modules provide ways to organize and manage collections of modules and parameters. They ensure proper registration for parameter discovery, state dict serialization, and device movement.

**Reference File:** `torch/nn/modules/container.py`

## Container Hierarchy

```
Container Modules
├── Sequential        - Ordered chain with automatic forward()
├── ModuleList        - List of modules (no automatic forward)
├── ModuleDict        - Dict of modules (no automatic forward)
├── ParameterList     - List of parameters
└── ParameterDict     - Dict of parameters
```

---

## Sequential

Ordered container that chains module outputs to inputs automatically.

### Class Definition

```python
class Sequential(Module):
    def __init__(self, *args: Module) -> None: ...
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...
```

### Key Behavior

- Modules added in order and executed sequentially
- Output of each module becomes input to the next
- Provides automatic `forward()` implementation

### Forward Implementation

```python
def forward(self, input):
    for module in self:
        input = module(input)
    return input
```

### Construction Patterns

```python
# Positional arguments (auto-numbered keys)
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)
# Keys: '0', '1', '2', '3'

# OrderedDict (named keys)
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 64, 5)),
    ('relu2', nn.ReLU())
]))
# Keys: 'conv1', 'relu1', 'conv2', 'relu2'
```

### List-like Operations

```python
seq = nn.Sequential(nn.Linear(10, 20), nn.ReLU())

# Indexing
layer = seq[0]           # Get first module
seq[0] = nn.Linear(10, 30)  # Replace module

# Slicing (returns new Sequential)
first_two = seq[:2]
last_two = seq[-2:]

# Length
len(seq)  # -> 2

# Iteration
for module in seq:
    print(module)

# Append
seq.append(nn.Linear(20, 30))

# Insert
seq.insert(1, nn.BatchNorm1d(20))

# Extend
seq.extend(nn.Sequential(nn.Linear(30, 40), nn.Sigmoid()))

# Pop
removed = seq.pop(0)  # Remove and return

# Delete
del seq[0]
del seq[1:3]  # Slice deletion

# Concatenation
seq1 + seq2  # Returns new Sequential
seq1 += seq2  # In-place extend

# Multiplication (repeat)
seq * 3  # Returns new Sequential with 3 copies
seq *= 2  # In-place repeat
```

### Use Case

Ideal for simple feed-forward networks without branching:

```python
classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

output = classifier(image_batch)  # Automatic chaining
```

---

## ModuleList

List container for modules. Does **not** provide automatic `forward()`.

### Class Definition

```python
class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] | None = None) -> None
```

### Key Difference from Sequential

- **ModuleList**: Just storage; you define how modules are used
- **Sequential**: Storage + automatic chaining in `forward()`

### List-like Operations

```python
layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

# Indexing
layer = layers[0]
layers[2] = nn.Linear(10, 20)

# Slicing (returns new ModuleList)
subset = layers[:3]

# Length
len(layers)  # -> 5

# Iteration
for layer in layers:
    print(layer)

# Append
layers.append(nn.Linear(10, 10))

# Insert
layers.insert(2, nn.BatchNorm1d(10))

# Extend
layers.extend([nn.Linear(10, 10), nn.ReLU()])

# Pop
removed = layers.pop(0)

# Concatenation
layers1 + layers2  # Returns new ModuleList
layers1 += layers2  # In-place extend
```

### Use Cases

**Shared iteration patterns:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(64) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer, norm in zip(self.layers, self.norms):
            x = x + F.relu(norm(layer(x)))
        return x
```

**Selective execution:**
```python
class MultiPathNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU()),
            nn.Sequential(nn.Conv2d(3, 64, 5), nn.ReLU()),
            nn.Sequential(nn.Conv2d(3, 64, 7), nn.ReLU()),
        ])

    def forward(self, x, path_idx):
        return self.paths[path_idx](x)  # Dynamic path selection
```

### Compressed Representation

ModuleList provides intelligent `__repr__` that compresses repeated modules:

```python
layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
print(layers)
# ModuleList(
#   (0-4): 5 x Linear(in_features=10, out_features=10, bias=True)
# )
```

---

## ModuleDict

Dictionary container for modules. Ordered and does **not** provide automatic `forward()`.

### Class Definition

```python
class ModuleDict(Module):
    def __init__(self, modules: Mapping[str, Module] | None = None) -> None
```

### Dictionary-like Operations

```python
choices = nn.ModuleDict({
    'conv': nn.Conv2d(3, 64, 3),
    'pool': nn.MaxPool2d(2),
    'fc': nn.Linear(64, 10)
})

# Indexing
layer = choices['conv']
choices['new'] = nn.BatchNorm2d(64)

# Length
len(choices)  # -> 4

# Iteration (over keys)
for key in choices:
    print(key)

# Containment
'conv' in choices  # -> True

# Keys, values, items
choices.keys()
choices.values()
choices.items()

# Update
choices.update({'extra': nn.Dropout(0.5)})

# Pop
removed = choices.pop('conv')

# Clear
choices.clear()
```

### Construction Patterns

```python
# From dict
modules = nn.ModuleDict({
    'conv': nn.Conv2d(3, 64, 3),
    'pool': nn.MaxPool2d(2)
})

# From list of pairs (preserves order)
modules = nn.ModuleDict([
    ['conv', nn.Conv2d(3, 64, 3)],
    ['pool', nn.MaxPool2d(2)]
])
```

### Use Cases

**Named module selection:**
```python
class ConfigurableNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        })
        self.linear = nn.Linear(64, 64)

    def forward(self, x, activation='relu'):
        x = self.linear(x)
        return self.activations[activation](x)
```

**Dynamic architecture:**
```python
class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(...)
        self.heads = nn.ModuleDict({
            'classification': nn.Linear(512, 10),
            'regression': nn.Linear(512, 1),
            'segmentation': nn.ConvTranspose2d(512, 21, 4)
        })

    def forward(self, x, task):
        features = self.backbone(x)
        return self.heads[task](features)
```

---

## ParameterList

List container for parameters. Converts Tensors to Parameters automatically.

### Class Definition

```python
class ParameterList(Module):
    def __init__(self, values: Iterable[Any] | None = None) -> None
```

### Key Behavior

- Tensors added are automatically wrapped in `Parameter`
- Parameters are properly registered for optimization
- Supports list-like operations

### Usage

```python
class MultiScaleWeights(nn.Module):
    def __init__(self, num_scales):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(1)) for _ in range(num_scales)
        ])

    def forward(self, features):
        # features: list of tensors at different scales
        return sum(w * f for w, f in zip(self.weights, features))
```

### Operations

```python
params = nn.ParameterList([
    nn.Parameter(torch.randn(10, 10)),
    nn.Parameter(torch.randn(20, 20))
])

# Indexing
p = params[0]
params[0] = torch.randn(15, 15)  # Auto-converted to Parameter

# Append
params.append(torch.randn(5, 5))  # Auto-converted to Parameter

# Extend
params.extend([torch.randn(3, 3), torch.randn(4, 4)])

# Iteration
for p in params:
    print(p.shape)

# Length
len(params)  # -> 5
```

---

## ParameterDict

Dictionary container for parameters. Ordered and converts Tensors automatically.

### Class Definition

```python
class ParameterDict(Module):
    def __init__(self, parameters: Any = None) -> None
```

### Key Behavior

- Tensors added are automatically wrapped in `Parameter`
- Parameters are properly registered for optimization
- Ordered dictionary semantics

### Usage

```python
class AttentionWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterDict({
            'query': nn.Parameter(torch.randn(64, 64)),
            'key': nn.Parameter(torch.randn(64, 64)),
            'value': nn.Parameter(torch.randn(64, 64))
        })

    def forward(self, x):
        q = x @ self.weights['query']
        k = x @ self.weights['key']
        v = x @ self.weights['value']
        return F.scaled_dot_product_attention(q, k, v)
```

### Operations

```python
params = nn.ParameterDict({
    'weight': nn.Parameter(torch.randn(10, 10)),
    'bias': nn.Parameter(torch.randn(10))
})

# Indexing
p = params['weight']
params['new'] = torch.randn(5, 5)  # Auto-converted to Parameter

# Update
params.update({'extra': torch.randn(3, 3)})

# Keys, values, items
params.keys()
params.values()
params.items()

# Get with default
p = params.get('missing', default=None)

# Pop
removed = params.pop('weight')

# Clear
params.clear()

# Copy
params_copy = params.copy()
```

---

## Registration Importance

### Why Use Containers vs Plain Python Lists/Dicts?

**Plain Python (NOT registered):**
```python
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10, 10) for _ in range(3)]  # NOT REGISTERED!

model = BadModel()
list(model.parameters())  # Empty! Parameters not found
model.to('cuda')  # Layers stay on CPU!
model.state_dict()  # Missing layer weights!
```

**Using ModuleList (properly registered):**
```python
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

model = GoodModel()
list(model.parameters())  # All 6 parameters found
model.to('cuda')  # All layers moved to GPU
model.state_dict()  # All weights serialized
```

---

## Comparison Table

| Container | Auto-forward | Use Case |
|-----------|--------------|----------|
| `Sequential` | Yes | Simple chains, no branching |
| `ModuleList` | No | Custom iteration patterns |
| `ModuleDict` | No | Named/dynamic module selection |
| `ParameterList` | N/A | Lists of learnable parameters |
| `ParameterDict` | N/A | Named learnable parameters |

---

## MLX Mapping

### Direct Equivalents

MLX uses a similar pattern with `mlx.nn.Module`:

```python
# MLX ModuleList pattern
class Model(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [mlx.nn.Linear(10, 10) for _ in range(3)]
        # MLX auto-registers lists of modules
```

### Key Differences

1. **Auto-registration**: MLX may auto-detect module lists without explicit `ModuleList`
2. **Functional style**: MLX often uses functional transformations instead of containers
3. **Parameter handling**: MLX parameters work differently with its functional approach

### Porting Example

```python
# PyTorch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# MLX equivalent
class Net(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [mlx.nn.Linear(64, 64) for _ in range(3)]

    def __call__(self, x):
        for layer in self.layers:
            x = mlx.nn.relu(layer(x))
        return x
```

---

## Best Practices

1. **Use Sequential** for simple linear chains
2. **Use ModuleList** when you need custom iteration or indexing
3. **Use ModuleDict** for dynamic/named module selection
4. **Always use containers** instead of plain Python lists/dicts for modules
5. **Use ParameterList/Dict** for learnable parameters that aren't part of submodules
