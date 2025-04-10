"""StringOperator - Arguments
(:mod:`qurry.qurries.string_operator.arguments`)

"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from .utils import AvailableStringOperatorTypes
from ...qurrium.experiment import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class StringOperatorArguments(ArgumentsPrototype):
    """Arguments for the experiment."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    num_qubits: int = 0
    """The number of qubits."""
    i: Optional[int] = None
    """The index of beginning qubits in the quantum circuit."""
    k: Optional[int] = None
    """The index of ending qubits in the quantum circuit."""
    str_op: AvailableStringOperatorTypes = "i"
    """The string operator."""


class StringOperatorMeasureArgs(BasicArgs, total=False):
    """Output arguments for :meth:`output`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    i: Optional[int]
    """The index of beginning qubits in the quantum circuit."""
    k: Optional[int]
    """The index of ending qubits in the quantum circuit."""
    str_op: AvailableStringOperatorTypes
    """The string operator."""


class StringOperatorOutputArgs(OutputArgs):
    """Output arguments for :meth:`output`."""

    i: Optional[int]
    """The index of beginning qubits in the quantum circuit."""
    k: Optional[int]
    """The index of ending qubits in the quantum circuit."""
    str_op: AvailableStringOperatorTypes
    """The string operator."""


class StringOperatorAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of the analyze method."""


SHORT_NAME = "qurries_string_operator"
