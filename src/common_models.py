import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from abc import ABC
from flax import linen as nn
from typing import Any, Callable, Dict

import chex
import numpy as np

from src.constants import *
from src.layer_modules import *


def get_activation(activation: str) -> Callable:
    """
    Gets an activation function

    :param activation: the activation function name
    :type activation: str
    :return: an activation function
    :rtype: Callable

    """
    assert (
        activation in VALID_ACTIVATION
    ), f"{activation} is not supported (one of {VALID_ACTIVATION})"

    if activation == CONST_IDENTITY:

        def identity(x: Any) -> Any:
            return x

        return identity
    elif activation == CONST_RELU:
        return nn.relu
    elif activation == CONST_TANH:
        return nn.tanh
    else:
        raise NotImplementedError


class Model(ABC):
    """Abstract model class."""

    #: Model forward call.
    forward: Callable

    #: Initialize model parameters.
    init: Callable

    def reset_h_state(self) -> chex.Array:
        """
        Resets hidden state.

        :return: a hidden state
        :rtype: chex.Array
        """
        return np.zeros((1,), dtype=np.float32)

    def update_batch_stats(
        self, params: Dict[str, Any], batch_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        return params

    @property
    def random_keys(self):
        return []
