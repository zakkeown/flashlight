import torch

# Test vector_norm
t = torch.randn(3, 4)
print('Testing vector_norm...')
try:
    result = torch.linalg.vector_norm(t)
    print('Success with positional')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.linalg.vector_norm(input=t)
    print('Success with input=')
except Exception as e:
    print(f'Error with input=: {e}')

try:
    result = torch.linalg.vector_norm(x=t)
    print('Success with x=')
except Exception as e:
    print(f'Error with x=: {e}')

# Test pinv
print('\nTesting pinv...')
try:
    result = torch.linalg.pinv(t)
    print('Success with positional')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.linalg.pinv(input=t)
    print('Success with input=')
except Exception as e:
    print(f'Error with input=: {e}')

try:
    result = torch.linalg.pinv(A=t)
    print('Success with A=')
except Exception as e:
    print(f'Error with A=: {e}')

# Test embedding
print('\nTesting embedding...')
weight = torch.randn(10, 5)
indices = torch.tensor([1, 2, 3])
try:
    result = torch.embedding(indices, weight)
    print('Success with (indices, weight)')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.embedding(weight, indices)
    print('Success with (weight, indices)')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.embedding(input=indices, weight=weight)
    print('Success with input=indices, weight=weight')
except Exception as e:
    print(f'Error: {e}')

# Test orgqr
print('\nTesting orgqr...')
a = torch.randn(5, 3)
tau = torch.randn(3)
try:
    result = torch.orgqr(a, tau)
    print('Success with (a, tau)')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.orgqr(input=a, tau=tau)
    print('Success with input=a, tau=tau')
except Exception as e:
    print(f'Error: {e}')

# Test ormqr
print('\nTesting ormqr...')
other = torch.randn(5, 2)
try:
    result = torch.ormqr(a, tau, other)
    print('Success with (a, tau, other)')
except Exception as e:
    print(f'Error: {e}')

try:
    result = torch.ormqr(input=a, tau=tau, other=other)
    print('Success with input=a, tau=tau, other=other')
except Exception as e:
    print(f'Error: {e}')
