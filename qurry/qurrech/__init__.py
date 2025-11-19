"""Qurrech - Wave Function Overlap (:mod:`qurry.qurrech`)"""

from typing import Literal, Union, overload

from .hadamard_test import EchoListenHadamard, EchoListenHadamardMeasureArgs
from .randomized_measure import EchoListenRandomized, EchoListenRandomizedMeasureArgs


# pylint: disable=invalid-name
@overload
def EchoListen(*args, method: Literal["hadamard"], **kwargs) -> EchoListenHadamard: ...


@overload
def EchoListen(
    *args, method: Union[Literal["randomized", "base"], str] = "randomized", **kwargs
) -> EchoListenRandomized: ...


def EchoListen(
    *args,
    method="randomized",
    **kwargs,
):
    """Call :func:`EchoListen` methods.

    Args:
        method (Literal["randomized", "hadamard"], optional):
            The method to use for wave function overlap measurement.

            - randomized: running by haar randomized measure.
            - hadamard: running by hadamard test.

            Defaults to 'randomized'.

    Returns:
        EchoListenBase: method.
    """
    if method == "hadamard":
        return EchoListenHadamard(*args, **kwargs)
    return EchoListenRandomized(*args, **kwargs)


@overload
def WaveFunctionOverlap(*args, method: Literal["hadamard"], **kwargs) -> EchoListenHadamard: ...


@overload
def WaveFunctionOverlap(
    *args, method: Union[Literal["randomized", "base"], str] = "randomized", **kwargs
) -> EchoListenRandomized: ...


def WaveFunctionOverlap(*args, method="randomized", **kwargs):
    """Call :func:`WaveFunctionOverlap` methods, another name of :func:`EchoListen`.

    Args:
        method (Literal["randomized", "hadamard"], optional):
            The method to use for wave function overlap measurement.

            - randomized: running by haar randomized measure.
            - hadamard: running by hadamard test.

            Defaults to 'randomized'.

    Returns:
        WaveFunctionOverlapBase: method.
    """
    if method == "hadamard":
        return EchoListenHadamard(*args, **kwargs)
    return EchoListenRandomized(*args, **kwargs)


__all__ = [
    "EchoListen",
    "EchoListenHadamard",
    "EchoListenHadamardMeasureArgs",
    "EchoListenRandomized",
    "EchoListenRandomizedMeasureArgs",
    "WaveFunctionOverlap",
]
