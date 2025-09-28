"""Qurrent/Qurshady - Second Renyi Entropy Measurement/Classical Shadow (:mod:`qurry.qurrent`)

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

from .randomized_measure import EntropyMeasureRandomized, EntropyMeasureRandomizedMeasureArgs
from .randomized_measure_v1 import EntropyMeasureRandomizedV1, EntropyMeasureRandomizedV1MeasureArgs
from .hadamard_test import EntropyMeasureHadamard, EntropyMeasureHadamardMeasureArgs
from .classical_shadow import ShadowUnveil, ShadowUnveilMeasureArgs


# pylint: disable=invalid-name
@overload
def EntropyMeasure(*args, method: Literal["hadamard"], **kwargs) -> EntropyMeasureHadamard: ...


@overload
def EntropyMeasure(
    *args, method: Literal["randomized_v1"], **kwargs
) -> EntropyMeasureRandomizedV1: ...


@overload
def EntropyMeasure(
    *args, method: Union[Literal["randomized", "haar", "base"], str] = "randomized", **kwargs
) -> EntropyMeasureRandomized: ...


@overload
def EntropyMeasure(*args, method: Literal["classical_shadow"], **kwargs) -> ShadowUnveil: ...


def EntropyMeasure(
    *args,
    method="randomized",
    **kwargs,
):
    """Call :func:`EntropyMeasure` methods.

    Args:
        method (Literal["randomized", "randomized_v1", "hadamard", "classical_shadow"], optional):
            The method to use for entropy measurement.

            - randomized: running by haar randomized measure.
            - hadamard: running by hadamard test.
            - base: the base of :class:`EntropyMeasure`.

            Defaults to 'randomized'.
    """
    if method in ("randomized", "haar"):
        return EntropyMeasureRandomized(*args, **kwargs)
    if method == "randomized_v1":
        return EntropyMeasureRandomizedV1(*args, **kwargs)
    if method == "hadamard":
        return EntropyMeasureHadamard(*args, **kwargs)
    if method == "classical_shadow":
        return ShadowUnveil(*args, **kwargs)
    return EntropyMeasureRandomized(*args, **kwargs)


# pylint: enable=invalid-name
