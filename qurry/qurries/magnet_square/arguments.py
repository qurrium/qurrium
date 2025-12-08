"""MagnetSquare - Arguments (:mod:`qurry.qurries.magnet_square.arguments`)"""

from typing import Optional, Union, Literal
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator

from ...qurrium import ArgumentsPrototype, BasicArgs, OutputArgs, WCKeyable


@dataclass(frozen=True)
class MSArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurries.magnet_square.experiment.MSExperiment`."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    num_qubits: int = 0
    """The number of qubits."""
    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]] = "z"
    """The unitary operator to apply.
    It can be a :class:`~qiskit.quantum_info.Operator`,
    a :class:`~qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). 
    Defaults to 'z'."""


class MSMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurries.magnet_square.qurry.MagnetSquare.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Optional[Union[QuantumCircuit, WCKeyable]]
    """The key or the circuit to execute."""
    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]]
    """The unitary operator to apply.
    It can be a :class:`~qiskit.quantum_info.Operator`,
    a :class:`~qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). 
    Defaults to 'z'."""


class MSOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurries.magnet_square.qurry.MagnetSquare.output`."""

    unitary_operator: Union[Operator, Gate, Literal["x", "y", "z"]]
    """The unitary operator to apply.
    It can be a :class:`~qiskit.quantum_info.Operator`,
    a :class:`~qiskit.circuit.Gate`, or a string
    representing the axis of rotation ('x', 'y', or 'z'). 
    Defaults to 'z'."""


SHORT_NAME = "qurmagsq_magnet_square"
"""The short name of :class:`~qurry.qurries.magnet_square.qurry.MagnetSquare`."""

ACRONYM = "MS"
"""The acronym of :class:`~qurry.qurries.magnet_square.qurry.MagnetSquare`."""
