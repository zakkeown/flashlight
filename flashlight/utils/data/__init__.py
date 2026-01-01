"""
torch.utils.data compatible module for MLX.

Re-exports all data loading components from flashlight.data for
PyTorch API compatibility.
"""

# Re-export everything from flashlight.data
from flashlight.data import (  # Datasets; Samplers; DataLoader; Utility functions; DataPipes; Validation and determinism; Internal
    BatchSampler,
    ChainDataset,
    ConcatDataset,
    DataChunk,
    DataLoader,
    Dataset,
    DFIterDataPipe,
    DistributedSampler,
    IterableDataset,
    IterDataPipe,
    MapDataPipe,
    RandomSampler,
    Sampler,
    SequentialSampler,
    StackDataset,
    Subset,
    SubsetRandomSampler,
    TensorDataset,
    WeightedRandomSampler,
    _DatasetKind,
    argument_validation,
    default_collate,
    default_convert,
    functional_datapipe,
    get_worker_info,
    guaranteed_datapipes_determinism,
    non_deterministic,
    random_split,
    runtime_validation,
    runtime_validation_disabled,
)

__all__ = [
    # Datasets
    "Dataset",
    "TensorDataset",
    "IterableDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset",
    "StackDataset",
    "random_split",
    # Samplers
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
    "DistributedSampler",
    # DataLoader
    "DataLoader",
    "default_collate",
    # Utility functions
    "get_worker_info",
    "default_convert",
    # DataPipes
    "IterDataPipe",
    "MapDataPipe",
    "DFIterDataPipe",
    "DataChunk",
    "functional_datapipe",
    # Validation and determinism
    "argument_validation",
    "runtime_validation",
    "runtime_validation_disabled",
    "guaranteed_datapipes_determinism",
    "non_deterministic",
    # Internal
    "_DatasetKind",
]
