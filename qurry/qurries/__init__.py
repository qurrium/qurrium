"""Qurries - All Qurrium Runtime Realization (:mod:`qurry.qurries`)

.. tip::

    1.  The name "qurrent" was the second proposed name for this package.
        It's not simply "current" with a "q" replacing the "c",
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

.. tip::

    How to name a new Qurrium Runtime Realization:

    - For the short name of a new Qurrium, please follow the rules below:
        1.  Start with the prefix "qurr", derived from "qurrium".
        2.  Add a suffix that reflects the core concept or method of the realization.
            For example:
            - "ent" for entropy-related realizations (e.g., qurrent for entropy measurements).
            - "ech" for echo-related realizations (e.g., qurrech for Loschmidt echo measurements).
        3.  Ensure the suffix is concise and indicative of the realization's purpose.
        4.  Combine the prefix and suffix to form a unique and descriptive short name.

    - For the :class:`~qurrium.qurrium.QurriumPrototype` subclass name,
        please follow the rules below:
        1.  Use a descriptive name that clearly indicates the realization's function.
            It's better for a combination of two words that reflect the purpose.
            In particular, a noun followed by a verb or gerund is preferred.
            For example:
            - "EntropyMeasure" for "entropy" "measure"
            - "EchoListen" for Loschmidt "echo" "listen"
                (Although it's just a name, not real sound here.)
            - "WaveFunctionOverlap" for "wave function" "overlap", the another name of "EchoListen"
            - "ShadowUnveil" for classical shadow realizations. (Unveil the classical shadow.)
        2.  Capitalize the first letter of each word in the name (CamelCase).
        3.  Ensure the name is unique within the Qurrium framework to avoid conflicts.

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
    *args, method: Union[Literal["randomized"], str] = "randomized", **kwargs
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
    *args, method: Union[Literal["randomized"], str] = "randomized", **kwargs
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
    *args, method: Union[Literal["randomized"], str] = "randomized", **kwargs
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
    "EntropyMeasureRandomized",
    "EntropyMeasureHadamard",
    "EntropyMeasure",
    "EchoListenHadamard",
    "EchoListenRandomized",
    "EchoListen",
    "WaveFunctionOverlap",
    "SamplingExecuter",
    "WavesExecuter",
    "MagnetSquare",
    "ZDirMagnetSquare",
    "StringOperator",
    "ShadowUnveil",
    "EMRMeasureArgs",
    "EMHMeasureArgs",
    "ELHMeasureArgs",
    "ELRMeasureArgs",
    "SEMeasureArgs",
    "WEMeasureArgs",
    "MSMeasureArgs",
    "ZMSMeasureArgs",
    "SOMeasureArgs",
    "SUMeasureArgs",
]
