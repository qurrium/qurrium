"""EchoListenHadamard - Arguments (:mod:`qurry.qurrech.hadamard_test.arguments`)"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium.experiment import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class EchoListenHadamardArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurrech.hadamard_test.experiment.EchoListenHadamardExperiment`."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    degree: Optional[tuple[int, int]] = None
    """The degree range."""


class EchoListenHadamardMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave1: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    wave2: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    degree: Union[int, tuple[int, int], None]
    """The degree range."""


class EchoListenHadamardOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard.output`."""

    degree: Union[int, tuple[int, int], None]
    """The degree range."""


class EchoListenHadamardAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
    and :meth:`~qurry.qurrech.hadamard_test.experiment.EchoListenHadamardExperiment.analyze`.

    The post-processing of Hadamard test does not need any input.
    """


SHORT_NAME = "qurrech_hadamard"
"""The short name of 
:class:`~qurry.qurrech.hadamard_test.experiment.EchoListenHadamardExperiment`."""
