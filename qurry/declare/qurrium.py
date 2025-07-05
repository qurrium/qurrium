"""Declaration - Arguments (:mod:`qurry.declare.qurrium`)"""

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
"""The type hint for passmanager argument in 
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.output`."""


class BasicArgs(TypedDict, total=False):
    """Basic input fields for
    :meth:`~qurry.qurrium.qurrium.QurriumPrototype.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

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
"""The type var of :class:`BasicArgs` for
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.measure`
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput"""

ConfigListType = Union[list[dict[str, Any]], list[_MA], list[Union[_MA, dict[str, Any]]]]
"""The generic type hint for the input of
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiBulid`
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`.
"""


class OutputArgs(BasicArgs):
    """Basic output arguments for
    :meth:`~qurry.qurrium.qurrium.QurriumPrototype.output`."""

    circuits: list[Union[QuantumCircuit, Hashable]]


_OA = TypeVar("_OA", bound=OutputArgs)
"""The type var of :class:`OutputArgs` 
for :meth:`~qurry.qurrium.qurrium.QurriumPrototype.output` 

:class:`OutputArgs` is also used for passing arguments in an standard format to
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.output` 
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`.
"""


class AnalyzeArgs(TypedDict):
    """Analysis input prototype."""


_RA = TypeVar("_RA", bound=AnalyzeArgs)
"""The type var of :class:`AnalyzeArgs` for
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.analyze`
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
"""

SpecificAnalsisArgs = Optional[dict[Hashable, Union[_RA, dict[str, Any], bool]]]
"""The type hint for :meth:`~qurry.qurrium.multimanager.multimanager.analyze` 
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
"""
