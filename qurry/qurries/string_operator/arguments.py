"""StringOperator - Arguments (:mod:`qurry.qurries.string_operator.arguments`)"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from .utils import StringOperatorLibType, StringOperatorDirection
from ...qurrium.experiment import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class StringOperatorArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurries.string_operator.experiment.StringOperatorExperiment`."""

    exp_name: str
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    num_qubits: int
    """The number of qubits."""
    i: int
    """The index of beginning qubits in the quantum circuit."""
    k: int
    """The index of ending qubits in the quantum circuit."""
    str_op: StringOperatorLibType = "i"
    """The string operator."""
    on_dir: StringOperatorDirection = "x"
    """The direction of the string operator, either 'x' or 'y'."""


class StringOperatorMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurries.string_operator.qurry.StringOperator.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    i: Optional[int]
    """The index of beginning qubits in the quantum circuit."""
    k: Optional[int]
    """The index of ending qubits in the quantum circuit."""
    str_op: StringOperatorLibType
    """The string operator."""
    on_dir: StringOperatorDirection
    """The direction of the string operator, either 'x' or 'y'."""


class StringOperatorOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurries.string_operator.qurry.StringOperator.output`."""

    i: Optional[int]
    """The index of beginning qubits in the quantum circuit."""
    k: Optional[int]
    """The index of ending qubits in the quantum circuit."""
    str_op: StringOperatorLibType
    """The string operator."""
    on_dir: StringOperatorDirection
    """The direction of the string operator, either 'x' or 'y'."""


class StringOperatorAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis` and
    :meth:`~qurry.qurries.string_operator.experiment.StringOperatorExperiment.analyze`.
    """


SHORT_NAME = "qurstrop_string_operator"
"""The short name of 
:class:`~qurry.qurries.string_operator.experiment.StringOperatorExperiment`."""
