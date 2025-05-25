"""Declaration - Arguments (:mod:`qurry.declare.qurrium`)

Arguments for :meth:`output` from :cls:`QurriumPrototype`
"""

from typing import Optional, Union, TypedDict, Any, Literal, TypeVar
from collections.abc import Hashable
from pathlib import Path
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler.passmanager import PassManager

from .run import RunArgsType
from .transpile import TranspileArgs


PassManagerType = Optional[Union[str, PassManager, tuple[str, PassManager]]]
"""The type hint for passmanager argument in :meth:`output` from :cls:`QurriumPrototype`."""


class BasicArgs(TypedDict, total=False):
    """Basic output arguments for :meth:`output`."""

    shots: int
    backend: Optional[Backend]
    exp_name: str
    run_args: RunArgsType
    transpile_args: Optional[TranspileArgs]
    passmanager: PassManagerType
    tags: Optional[tuple[str, ...]]
    # already built exp
    exp_id: Optional[str]
    new_backend: Optional[Backend]
    revive: bool
    replace_circuits: bool
    # process tool
    qasm_version: Literal["qasm2", "qasm3"]
    export: bool
    save_location: Optional[Union[Path, str]]
    pbar: Optional[tqdm.tqdm]


_MA = TypeVar("_MA", bound=BasicArgs)
ConfigListType = Union[list[dict[str, Any]], list[_MA], list[Union[_MA, dict[str, Any]]]]
"""The type hint for :cls:`MultiManager` and 
:meth:`multiBulid`, :meth:`multiOutput` from :cls:`QurriumPrototype`.
"""


class OutputArgs(BasicArgs):
    """Basic output arguments for :meth:`output`."""

    circuits: list[Union[QuantumCircuit, Hashable]]


_OA = TypeVar("_OA", bound=OutputArgs)
"""The type hint for :meth:`measure_to_output` from :cls:`QurriumPrototype`.
OutputArgs is used for passing arguments in an standard format to
:meth:`output` from :cls:`QurriumPrototype` and 
:meth:`multiOutput` from :cls:`MultiManager`.
"""


class AnalyzeArgs(TypedDict):
    """Analysis input prototype."""


_RA = TypeVar("_RA", bound=AnalyzeArgs)
SpecificAnalsisArgs = Optional[dict[Hashable, Union[_RA, dict[str, Any], bool]]]
"""The type hint for :meth:`analyze` from :cls:`MultiManager`
and :meth:`multiAnalsis` from :cls:`QurriumPrototype`.
"""
