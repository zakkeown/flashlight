"""
Quantization Utilities

Provides functions for preparing and converting models for quantization.
"""

from typing import Callable, Dict, Optional, Set, Type

from ..nn.module import Module
from ..tensor import Tensor
from .observer import MinMaxObserver, ObserverBase
from .qconfig import QConfig, default_qat_qconfig, default_qconfig
from .fake_quantize import FakeQuantize


# Mapping from float modules to their quantized versions
DEFAULT_MODULE_MAPPING: Dict[Type[Module], Type[Module]] = {}

# Modules that support quantization
QUANTIZABLE_MODULES: Set[Type[Module]] = set()


def _get_qconfig_for_module(module: Module, qconfig_mapping: Optional[Dict] = None) -> Optional[QConfig]:
    """Get the QConfig for a module."""
    if qconfig_mapping is not None:
        # Check for exact match first
        if type(module) in qconfig_mapping:
            return qconfig_mapping[type(module)]
        # Check for name-based mapping
        for name, qconfig in qconfig_mapping.items():
            if isinstance(name, str) and hasattr(module, "name") and module.name == name:
                return qconfig
    # Return module's qconfig if it has one
    if hasattr(module, "qconfig"):
        return module.qconfig
    return None


def prepare(
    model: Module,
    qconfig_mapping: Optional[Dict] = None,
    example_inputs: Optional[Tensor] = None,
    inplace: bool = False,
) -> Module:
    """
    Prepare a model for post-training quantization (PTQ).

    Inserts observers into the model to collect statistics during calibration.

    Args:
        model: The model to prepare
        qconfig_mapping: Mapping from module types/names to QConfigs
        example_inputs: Example inputs for tracing (optional)
        inplace: If True, modify the model in place

    Returns:
        Prepared model with observers

    Example:
        >>> model = MyModel()
        >>> prepared = prepare(model)
        >>> # Run calibration data through the model
        >>> for batch in calibration_loader:
        ...     prepared(batch)
        >>> # Convert to quantized model
        >>> quantized = convert(prepared)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    # Use default qconfig if none provided
    if qconfig_mapping is None:
        qconfig_mapping = {"": default_qconfig}

    # Add observers to modules
    _add_observers(model, qconfig_mapping)

    return model


def _add_observers(module: Module, qconfig_mapping: Dict) -> None:
    """Recursively add observers to modules."""
    qconfig = _get_qconfig_for_module(module, qconfig_mapping)

    # Add activation observer if this module has qconfig
    if qconfig is not None and qconfig.activation is not None:
        observer = qconfig.activation()
        module._activation_post_process = observer

    # Add weight observer for supported modules
    if qconfig is not None and qconfig.weight is not None:
        if hasattr(module, "weight") and module.weight is not None:
            observer = qconfig.weight()
            observer(module.weight)  # Observe the weight
            module._weight_observer = observer

    # Recurse into children
    for name, child in module.named_children():
        _add_observers(child, qconfig_mapping)


def convert(
    model: Module,
    mapping: Optional[Dict[Type[Module], Type[Module]]] = None,
    inplace: bool = False,
) -> Module:
    """
    Convert a prepared model to a quantized model.

    Replaces float modules with their quantized counterparts using
    the collected statistics from observers.

    Args:
        model: Prepared model with observers
        mapping: Mapping from float to quantized module types
        inplace: If True, modify the model in place

    Returns:
        Quantized model

    Example:
        >>> prepared = prepare(model)
        >>> # ... run calibration ...
        >>> quantized = convert(prepared)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    if mapping is None:
        mapping = DEFAULT_MODULE_MAPPING

    # Convert modules
    _convert_modules(model, mapping)

    return model


def _convert_modules(module: Module, mapping: Dict[Type[Module], Type[Module]]) -> None:
    """Recursively convert modules to quantized versions."""
    # Quantize weights if observer exists
    if hasattr(module, "_weight_observer") and hasattr(module, "weight"):
        observer = module._weight_observer
        scale, zero_point = observer.calculate_qparams()

        # For now, we just store the quantization parameters
        # A full implementation would convert the weight to QuantizedTensor
        module._weight_scale = scale
        module._weight_zero_point = zero_point

    # Remove observers after conversion
    if hasattr(module, "_activation_post_process"):
        delattr(module, "_activation_post_process")
    if hasattr(module, "_weight_observer"):
        delattr(module, "_weight_observer")

    # Recurse into children
    for name, child in module.named_children():
        _convert_modules(child, mapping)


def prepare_qat(
    model: Module,
    qconfig_mapping: Optional[Dict] = None,
    example_inputs: Optional[Tensor] = None,
    inplace: bool = False,
) -> Module:
    """
    Prepare a model for quantization-aware training (QAT).

    Inserts FakeQuantize modules that simulate quantization during training.

    Args:
        model: The model to prepare for QAT
        qconfig_mapping: Mapping from module types/names to QConfigs
        example_inputs: Example inputs for tracing (optional)
        inplace: If True, modify the model in place

    Returns:
        Model prepared for QAT

    Example:
        >>> model = MyModel()
        >>> qat_model = prepare_qat(model)
        >>> # Train the model
        >>> for batch in train_loader:
        ...     loss = criterion(qat_model(batch), target)
        ...     loss.backward()
        ...     optimizer.step()
        >>> # Convert to quantized model
        >>> quantized = convert(qat_model)
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    # Use default QAT qconfig if none provided
    if qconfig_mapping is None:
        qconfig_mapping = {"": default_qat_qconfig}

    # Add FakeQuantize modules
    _add_fake_quantize(model, qconfig_mapping)

    return model


def _add_fake_quantize(module: Module, qconfig_mapping: Dict) -> None:
    """Recursively add FakeQuantize modules."""
    qconfig = _get_qconfig_for_module(module, qconfig_mapping)

    # Add activation fake quantize
    if qconfig is not None and qconfig.activation is not None:
        fake_quant = qconfig.activation()
        if isinstance(fake_quant, FakeQuantize):
            module._activation_fake_quantize = fake_quant

    # Add weight fake quantize for supported modules
    if qconfig is not None and qconfig.weight is not None:
        if hasattr(module, "weight") and module.weight is not None:
            fake_quant = qconfig.weight()
            if isinstance(fake_quant, FakeQuantize):
                module._weight_fake_quantize = fake_quant

    # Recurse into children
    for name, child in module.named_children():
        _add_fake_quantize(child, qconfig_mapping)


def fuse_modules(
    model: Module,
    modules_to_fuse: list,
    inplace: bool = False,
) -> Module:
    """
    Fuse a list of modules into a single module.

    Common fusions:
    - Conv + BatchNorm -> Conv
    - Conv + BatchNorm + ReLU -> ConvReLU
    - Linear + ReLU -> LinearReLU

    Args:
        model: The model containing modules to fuse
        modules_to_fuse: List of module name lists to fuse
        inplace: If True, modify the model in place

    Returns:
        Model with fused modules

    Note:
        Module fusion is not fully implemented. This is a placeholder
        for API compatibility.
    """
    import warnings
    warnings.warn(
        "Module fusion is not fully implemented in flashlight. "
        "The model will be returned unchanged.",
        UserWarning,
    )

    if not inplace:
        import copy
        model = copy.deepcopy(model)

    return model


def get_observer_dict(model: Module) -> Dict[str, ObserverBase]:
    """
    Get all observers from a prepared model.

    Args:
        model: Prepared model with observers

    Returns:
        Dictionary mapping module names to their observers
    """
    observers = {}

    def _collect_observers(module: Module, prefix: str = "") -> None:
        name = prefix if prefix else "root"

        if hasattr(module, "_activation_post_process"):
            observers[f"{name}.activation"] = module._activation_post_process
        if hasattr(module, "_weight_observer"):
            observers[f"{name}.weight"] = module._weight_observer

        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            _collect_observers(child, child_prefix)

    _collect_observers(model)
    return observers
