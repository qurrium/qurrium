"""Qurrium - The Quantum Experiment Manager for Qiskit
and The Measuring Tool for Renyi Entropy, Loschmidt Echo, and More

"""

import sys

from . import boorust

# pylint: disable=c-extension-no-member,wrong-import-position

# Due to PyO3 submodule is not fully compatible as a Python module.
# So we will need to assign them manually like the following does.
# Qiskit also does something similar which I 'learned' from them at beginning.
# They do not create pyi files for their Rust binding, but I made it here.
sys.modules["qurry.boorust.counts_process"] = boorust.counts_process  # type: ignore
sys.modules["qurry.boorust.bit_slice"] = boorust.bit_slice  # type: ignore
sys.modules["qurry.boorust.randomized"] = boorust.randomized  # type: ignore
sys.modules["qurry.boorust.hadamard"] = boorust.hadamard  # type: ignore
sys.modules["qurry.boorust.magnet_square"] = boorust.magnet_square  # type: ignore
sys.modules["qurry.boorust.string_operator"] = boorust.string_operator  # type: ignore
sys.modules["qurry.boorust.dummy"] = boorust.dummy  # type: ignore
sys.modules["qurry.boorust.test"] = boorust.test  # type: ignore


from .qurrech import EchoListen, WaveFunctionOverlap
from .qurrent import EntropyMeasure, ShadowUnveil
from .qurries import WavesExecuter, SamplingExecuter, MagnetSquare, ZDirMagnetSquare, StringOperator
from .tools import (
    BackendWrapper,
    version_check,
    cmd_wrapper,
    pytorch_cuda_check,
    fun_platform_check,
)
from .process.randomized_measure import generate_random_unitary_seeds, check_random_unitary_seeds
from .process.classical_shadow import generate_random_basis, check_random_basis
from .process.availability import availablility
from .version import __version__


BACKEND_AVAILABLE = availablility("boorust", [("Rust", True, None)])
