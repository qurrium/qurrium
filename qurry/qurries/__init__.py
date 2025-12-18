"""Qurries - All Qurrium Runtime Realization (:mod:`qurry.qurries`)

.. tip::

    1.  The name "qurrent" was the second proposed name for this package.
        It’s not simply "current" with a "q" replacing the "c",
        but rather a combination of "qurr" and "ent", where "ent" abbreviates "entropy",
        and "qurr" is a coined prefix beginning with "qu" to imply "quantum".

    2.  The initial proposed name for the package was "xproc", short for "experimental process".

    3.  Another module, "qurrech", combines "qurr" with "ech",
        where "ech" is short for "echo",
        the Loschmidt echo being a key concept in quantum information theory.
        This inspired us to adopt "qurry", derived from "qurr" and the suffix "ry",
        to resemble words like "query" or "curry".

    4.  Since a package named "qurry" already exists on PyPI,
        and many projects on GitHub also use this name,
        we changed the suffix from "y" to "ium", creating "qurrium".
        "Qurrium" is unique and easily searchable,
        with no prior use on PyPI or in general web searches.

    5.  So there is the evolution of the package name:
        xproc -> qurrent -> qurry -> qurrium

    6. Emoji: Qurry 🍛 / Qurrium 📏

"""

from typing import Literal, Union, overload

from .entropy_randomized import EntropyMeasureRandomized, EMRMeasureArgs
from .entropy_hadamard import EntropyMeasureHadamard, EMHMeasureArgs
from .echo_hadamard import EchoListenHadamard, ELHMeasureArgs
from .echo_randomized import EchoListenRandomized, ELRMeasureArgs
from .samplingqurry import SamplingExecuter, SEMeasureArgs
from .wavesqurry import WavesExecuter, WEMeasureArgs
from .magnet_square import MagnetSquare, MSMeasureArgs
from .magnet_square_z import ZDirMagnetSquare, ZMSMeasureArgs
from .string_operator import StringOperator, SOMeasureArgs
from .classical_shadow import ShadowUnveil, SUMeasureArgs

# from .twisted_operator import TwistedOperator
# from .position_distribution import PositionDistribution


# pylint: disable=invalid-name
@overload
def EntropyMeasure(*args, method: Literal["hadamard"], **kwargs) -> EntropyMeasureHadamard: ...
@overload
def EntropyMeasure(
    *args, method: Union[Literal["randomized"], str], **kwargs
) -> EntropyMeasureRandomized: ...


def EntropyMeasure(*args, method="randomized", **kwargs):
    """Call :func:`EntropyMeasure` methods.

    Args:
        method (Literal["randomized", "hadamard"], optional):
            The method to use for entropy measurement.

            - randomized: running by haar randomized measure.
            - hadamard: running by hadamard test.
            - base: the base of :class:`EntropyMeasure`.

            Defaults to 'randomized'.
    """
    if method == "hadamard":
        return EntropyMeasureHadamard(*args, **kwargs)
    return EntropyMeasureRandomized(*args, **kwargs)


@overload
def EchoListen(*args, method: Literal["hadamard"], **kwargs) -> EchoListenHadamard: ...
@overload
def EchoListen(
    *args, method: Union[Literal["randomized"], str], **kwargs
) -> EchoListenRandomized: ...


def EchoListen(*args, method="randomized", **kwargs):
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
    *args, method: Union[Literal["randomized"], str], **kwargs
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
