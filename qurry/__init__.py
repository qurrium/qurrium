"""Qurry 🍛 - The Quantum Experiment Manager for Qiskit
and The Measuring Tool for Renyi Entropy, Loschmidt Echo, and More

"""

import sys

from .qurrech import EchoListen, WaveFunctionOverlap
from .qurrent import EntropyMeasure, ShadowUnveil

# from .qurmagsq import MagnetSquare
# from .qurstrop import StringOperator

from .qurrium import WavesExecuter, SamplingExecuter

from .tools import (
    BackendWrapper,
    version_check,
    cmd_wrapper,
    pytorch_cuda_check,
    fun_platform_check,
)

from .version import __version__

# pylint: disable=no-name-in-module,import-error
# pylint: disable=wrong-import-order,no-member,c-extension-no-member

# But Qiskit found a solution to this problem.
# You can find this implementation in the Qiskit source code.
# In their root __init__.py file,
# they import the submodules and assign them to the sys.modules dictionary.
try:
    import qurry.boorust  # type: ignore

    sys.modules["qurry.boorust.counts_process"] = qurry.boorust.counts_process  # type: ignore
    sys.modules["qurry.boorust.bit_slice"] = qurry.boorust.bit_slice  # type: ignore
    sys.modules["qurry.boorust.randomized"] = qurry.boorust.randomized  # type: ignore
    sys.modules["qurry.boorust.hadamard"] = qurry.boorust.hadamard  # type: ignore
    sys.modules["qurry.boorust.dummy"] = qurry.boorust.dummy  # type: ignore
    sys.modules["qurry.boorust.test"] = qurry.boorust.test  # type: ignore
    # sys.modules["qurry.boorust.shadow"] = qurry.boorust.shadow  # type: ignore
    RUST_AVAILABLE = True
    FAILED_RUST_IMPORT = None
except ModuleNotFoundError as qurry_boorust_import_error:
    RUST_AVAILABLE = False
    FAILED_RUST_IMPORT = qurry_boorust_import_error

from .process.availability import availablility

BACKEND_AVAILABLE = availablility(
    "boorust",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)
# pylint: enable=no-name-in-module,import-error
# pylint: enable=wrong-import-order,no-member,c-extension-no-member

# """
# Note that this does not define a package,
# so this won’t allow Python code to directly import submodules
# by using from parent_module import child_module.
# For more information,
# see [#759](https://github.com/PyO3/pyo3/issues/759)
# and [#1517](https://github.com/PyO3/pyo3/issues/1517).
# from https://pyo3.rs/v0.23.0/module.html#python-submodules
# (Since PyO3 0.20.0, until PyO3 0.23.0)
# :smile:
# """

# DO NOT MAKE .pyi FOR boorust MODULES.
# it will overwrite and corrupt the module from rust.
