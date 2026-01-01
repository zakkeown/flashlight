"""
Optimizer Base Class

Implements the base class for all optimizers, following PyTorch's design.
"""

from typing import Iterable, Dict, Any, Optional, Union, List
from collections import defaultdict
from ..nn.parameter import Parameter


class Optimizer:
    """
    Base class for all optimizers.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        defaults: Dict containing default values of optimization options

    Attributes:
        param_groups: List of parameter groups (each group is a dict)
        state: Dict containing optimizer state (momentum buffers, etc.)
    """

    def __init__(self, params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]], defaults: Dict[str, Any]):
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        # Handle both param iterables and param groups
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # Check if first element is a dict (param group) or Parameter
        if not isinstance(param_groups[0], dict):
            # Simple case: just a list of parameters
            param_groups = [{'params': param_groups}]

        # Process each param group
        for param_group in param_groups:
            self.add_param_group(param_group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer's param_groups.

        Args:
            param_group: Dict containing 'params' and other optimizer-specific options
        """
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")

        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)

        # Check for duplicate parameters
        param_set = set()
        for group in self.param_groups:
            param_set.update(id(p) for p in group['params'])

        for param in param_group['params']:
            if id(param) in param_set:
                raise ValueError("some parameters appear in more than one parameter group")

        # Apply defaults
        for name, default in self.defaults.items():
            if name != 'params':
                param_group.setdefault(name, default)

        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Set gradients of all optimized parameters to zero.

        Args:
            set_to_none: If True, set gradients to None instead of zero
                        (more memory efficient, default True)
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state of the optimizer as a dict.

        Returns:
            Dict containing:
                - state: optimizer state (momentum buffers, etc.)
                - param_groups: parameter groups with their options
        """
        # Map parameter ids to indices
        param_to_idx = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_to_idx[id(param)] = (group_idx, param_idx)

        # Pack state using parameter indices instead of ids
        packed_state = {}
        for param_id, state in self.state.items():
            if param_id in param_to_idx:
                packed_state[param_to_idx[param_id]] = state

        # Pack param_groups (excluding actual parameter tensors)
        param_groups = []
        for group in self.param_groups:
            packed_group = {k: v for k, v in group.items() if k != 'params'}
            packed_group['params'] = [param_to_idx[id(p)] for p in group['params']]
            param_groups.append(packed_group)

        return {
            'state': packed_state,
            'param_groups': param_groups
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the optimizer state.

        Args:
            state_dict: Optimizer state dict (from state_dict())
        """
        # Create mapping from indices to parameters
        idx_to_param = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                idx_to_param[(group_idx, param_idx)] = param

        # Unpack state
        self.state = defaultdict(dict)
        for idx, state in state_dict['state'].items():
            if idx in idx_to_param:
                param = idx_to_param[idx]
                self.state[id(param)] = state

        # Unpack param_groups (hyperparameters only, not params)
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for k, v in saved_group.items():
                if k != 'params':
                    group[k] = v

    def __repr__(self) -> str:
        """String representation of the optimizer."""
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string


__all__ = ['Optimizer']
