"""Declaration of the input fields of QurriumPrototype (:mod:`qurry.qurrium.utils.declare`)"""

from typing import Optional, Union, TypedDict, Any, Literal, TypeVar
from pathlib import Path
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .waves import WCKeyable
from .transpiler import PassManagerType, TranspileArgs


class BaseRunArgs(TypedDict):
    """Arguments for :meth:`~qiskit.providers.backend.BackendV2.run`."""


RunArgsType = Optional[Union[BaseRunArgs, dict[str, Any]]]
"""The type hint for :meth:`~qiskit.providers.backend.BackendV2.run`."""


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

    circuits: list[Union[QuantumCircuit, WCKeyable]]


_OA = TypeVar("_OA", bound=OutputArgs)
"""The type var of :class:`OutputArgs` 
for :meth:`~qurry.qurrium.qurrium.QurriumPrototype.output` 

:class:`OutputArgs` is also used for passing arguments in an standard format to
:meth:`~qurry.qurrium.qurrium.QurriumPrototype.output` 
and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`.
"""
