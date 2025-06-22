"""MagnetSquare - Arguments (:mod:`qurry.qurries.magnet_square.arguments`)"""

from typing import Optional, Union, Literal
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator

from ...qurrium.experiment import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class MagnetSquareArguments(ArgumentsPrototype):
    """Arguments for the experiment."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    num_qubits: int = 0
    """The number of qubits."""
    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]] = "z"
    """The unitary operator to apply.
    It can be a `qiskit.quantum_info.Operator`, a `qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). Defaults to 'z'."""


class MagnetSquareMeasureArgs(BasicArgs, total=False):
    """Output arguments for :meth:`output`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]]
    """The unitary operator to apply.
    It can be a `qiskit.quantum_info.Operator`, a `qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). Defaults to 'z'."""


class MagnetSquareOutputArgs(OutputArgs):
    """Output arguments for :meth:`output`."""

    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]]
    """The unitary operator to apply.
    It can be a `qiskit.quantum_info.Operator`, a `qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). Defaults to 'z'."""


class MagnetSquareAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of the analyze method."""


SHORT_NAME = "qurmagsq_magnet_square"
