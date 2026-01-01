"""
Optimizers and Learning Rate Schedulers

Implements PyTorch-compatible optimizers and LR scheduling strategies.
"""

from .optimizer import Optimizer
from .sgd import SGD
from .adam import Adam
from .adamw import AdamW
from .rmsprop import RMSprop
from .adagrad import Adagrad
from .adadelta import Adadelta
from .additional import Adamax, RAdam, NAdam, ASGD, Rprop
from .lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
)
from . import lr_scheduler

__all__ = [
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'Adamax',
    'RAdam',
    'NAdam',
    'ASGD',
    'Rprop',
    'StepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'ReduceLROnPlateau',
    'LinearLR',
    'lr_scheduler',
]
