from typing import Literal
import warnings

from ..qurrium.exceptions import UnconfiguredWarning
# v4
from .qurrentHaarV4 import EchoHaarMeasureV4
# v3
from .qurrech import EchoListen as EchoListenBase
from .haarMeasure import haarMeasure
from .hadamardTest import hadamardTest


def EchoListen(
    *args,
    method: Literal['randomized', 'hadamard', 'base'] = 'randomized',
    version: Literal['v4', 'v3'] = 'v3',
    **kwargs,
) -> EchoListenBase:
    """Call `EchoListen` methods.

    Args:
        method (Literal[&#39;randomized&#39;, &#39;hadamard&#39;, &#39;base&#39;], optional): 

            - randomized: running by haar randomized measure.
            - hadamard: running by hadamard test.
            - base: the base of `EchoListen`.
            Defaults to 'randomized'.

    Returns:
        EchoListenBase: method.
    """
    if version == 'v4':
        if method == 'hadamard':
            return hadamardTest(*args, **kwargs)
        else:
            return EchoHaarMeasureV4(*args, **kwargs)
    else:
        if method == 'base':
            warnings.warn(
                "This method is a base of 'EchoListen' which cannot work before" +
                " introduce measurement like haar randomized measure.", UnconfiguredWarning)
            return EchoListenBase(*args, **kwargs)
        elif method == 'hadamard':
            return hadamardTest(*args, **kwargs)
        else:
            return haarMeasure(*args, **kwargs)
