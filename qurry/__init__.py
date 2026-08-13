"""Qurrium - The Quantum Experiment Manager for Qiskit
and The Measuring Tool for Renyi Entropy, Loschmidt Echo, and More

"""

import importlib.util
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from . import boorust as _boorust


def _load_boorust_extension():
    """Load boorust extension module from installed site-packages when available."""
    current_package_dir = Path(__file__).resolve().parent

    for search_root in (Path(p) for p in sys.path if isinstance(p, str) and p):
        candidate_dir = search_root / "qurry"
        if not candidate_dir.exists():
            continue
        if candidate_dir.resolve() == current_package_dir:
            continue

        for suffix in EXTENSION_SUFFIXES:
            candidate = candidate_dir / f"boorust{suffix}"
            if not candidate.exists():
                continue

            spec = importlib.util.spec_from_file_location("qurry.boorust", candidate)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                continue

            module = importlib.util.module_from_spec(spec)
            previous_module = sys.modules.get("qurry.boorust")
            sys.modules["qurry.boorust"] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                if previous_module is None:
                    sys.modules.pop("qurry.boorust", None)
                else:
                    sys.modules["qurry.boorust"] = previous_module
                continue
            return module

    return _boorust


def _is_boorust_extension(module):
    module_file = getattr(module, "__file__", "") or ""
    return any(module_file.endswith(suffix) for suffix in EXTENSION_SUFFIXES)


boorust = _boorust if _is_boorust_extension(_boorust) else _load_boorust_extension()

# Due to PyO3 submodule is not fully compatible as a Python module.
# So we will need to assign them manually like the following does.
# Qiskit also does something similar which I 'learned' from them at beginning.
# They do not create pyi files for their Rust binding, but I made it here.
for boorust_submodule in (
    "counts_process",
    "bit_slice",
    "randomized",
    "hadamard",
    "magnet_square",
    "string_operator",
    "shadow",
    "dummy",
):
    boorust_module = getattr(boorust, boorust_submodule, None)
    if boorust_module is not None:
        sys.modules[f"qurry.boorust.{boorust_submodule}"] = boorust_module  # type: ignore

from .qurries import (
    EntropyMeasure,
    EchoListen,
    WaveFunctionOverlap,
    WavesExecuter,
    SamplingExecuter,
    MagnetSquare,
    ZDirMagnetSquare,
    StringOperator,
    ShadowUnveil,
)
from .tools import (
    get_qiskit_version_statesheet,
    cmd_wrapper,
    pytorch_cuda_check,
    fun_platform_check,
)
from .process.randomized_measure import generate_random_unitary_seeds, check_random_unitary_seeds
from .process.classical_shadow import generate_random_basis, check_random_basis
from .process.availability import availability
from .version import __version__


BACKEND_AVAILABLE = availability("boorust", [("Rust", True, None)])
