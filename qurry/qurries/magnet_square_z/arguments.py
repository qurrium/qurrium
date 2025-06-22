"""ZDirMagnetSquare - Arguments (:mod:`qurry.qurries.magnet_square_z.arguments`)"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium.experiment import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class ZDirMagnetSquareArguments(ArgumentsPrototype):
    """Arguments for the experiment."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    """
    num_qubits: int = 0
    """The number of qubits."""


class ZDirMagnetSquareMeasureArgs(BasicArgs, total=False):
    """Output arguments for :meth:`output`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""


class ZDirMagnetSquareOutputArgs(OutputArgs):
    """Output arguments for :meth:`output`."""


class ZDirMagnetSquareAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of the analyze method."""


SHORT_NAME = "qurmagsq_magnet_square_zdir"
