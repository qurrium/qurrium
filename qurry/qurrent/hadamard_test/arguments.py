"""EntropyMeasureHadamard - Arguments (:mod:`qurry.qurrent.hadamard_test.arguments`)"""

from typing import Optional, Union
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium import ArgumentsPrototype, BasicArgs, OutputArgs, WCKeyable


@dataclass(frozen=True)
class EMHArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurrent.hadamard_test.experiment.EntropyMeasureHadamard`."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    degree: Optional[tuple[int, int]] = None
    """The degree range."""


class EMHMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrent.hadamard_test.qurry.EchoListenHadamard.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Optional[Union[QuantumCircuit, WCKeyable]]
    """The key or the circuit to execute."""
    degree: Optional[Union[int, tuple[int, int]]]
    """The degree range."""


class EMHOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurrent.hadamard_test.qurry.EchoListenHadamard.output`."""

    degree: Optional[Union[int, tuple[int, int]]]
    """The degree range."""


SHORT_NAME = "qurrent_hadamard"
"""The short name of 
:class:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard`."""

ACRONYM = "EMH"
"""The abbreviation of
:class:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard`."""
