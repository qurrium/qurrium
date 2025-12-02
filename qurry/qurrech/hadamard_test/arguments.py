"""EchoListenHadamard - Arguments (:mod:`qurry.qurrech.hadamard_test.arguments`)"""

from typing import Optional, Union
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium import ArgumentsPrototype, BasicArgs, OutputArgs, WCKeyable


@dataclass(frozen=True)
class ELHArguments(ArgumentsPrototype):
    """Arguments for :class:`~qurry.qurrech.hadamard_test.experiment.ELHExperiment`."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    degree: Optional[tuple[int, int]] = None
    """The degree range."""


class ELHMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave1: Optional[Union[QuantumCircuit, WCKeyable]]
    """The key or the circuit to execute."""
    wave2: Optional[Union[QuantumCircuit, WCKeyable]]
    """The key or the circuit to execute."""
    degree: Union[int, tuple[int, int], None]
    """The degree range."""


class ELHOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard.output`."""

    degree: Union[int, tuple[int, int], None]
    """The degree range."""


SHORT_NAME = "qurrech_hadamard"
"""The short name of :class:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard`."""

ACRONYM = "ELH"
"""The abbreviation of :class:`~qurry.qurrech.hadamard_test.qurry.EchoListenHadamard`."""
