"""
torch.utils.data compatible module for MLX.

Re-exports all data loading components from mlx_compat.data for
PyTorch API compatibility.
"""

# Re-export everything from mlx_compat.data
from mlx_compat.data import (
    # Datasets
    Dataset,
    TensorDataset,
    IterableDataset,
    ConcatDataset,
    ChainDataset,
    Subset,
    StackDataset,
    random_split,
    # Samplers
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
    DistributedSampler,
    # DataLoader
    DataLoader,
    default_collate,
    # Utility functions
    get_worker_info,
    default_convert,
    # DataPipes
    IterDataPipe,
    MapDataPipe,
    DFIterDataPipe,
    DataChunk,
    functional_datapipe,
    # Validation and determinism
    argument_validation,
    runtime_validation,
    runtime_validation_disabled,
    guaranteed_datapipes_determinism,
    non_deterministic,
    # Internal
    _DatasetKind,
)

__all__ = [
    # Datasets
    'Dataset',
    'TensorDataset',
    'IterableDataset',
    'ConcatDataset',
    'ChainDataset',
    'Subset',
    'StackDataset',
    'random_split',
    # Samplers
    'Sampler',
    'SequentialSampler',
    'RandomSampler',
    'BatchSampler',
    'SubsetRandomSampler',
    'WeightedRandomSampler',
    'DistributedSampler',
    # DataLoader
    'DataLoader',
    'default_collate',
    # Utility functions
    'get_worker_info',
    'default_convert',
    # DataPipes
    'IterDataPipe',
    'MapDataPipe',
    'DFIterDataPipe',
    'DataChunk',
    'functional_datapipe',
    # Validation and determinism
    'argument_validation',
    'runtime_validation',
    'runtime_validation_disabled',
    'guaranteed_datapipes_determinism',
    'non_deterministic',
    # Internal
    '_DatasetKind',
]
