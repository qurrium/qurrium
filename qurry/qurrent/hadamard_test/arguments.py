"""EntropyMeasureHadamard - Arguments (:mod:`qurry.qurrent.hadamard_test.arguments`)"""

from typing import Any, Union
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium import ArgumentsPrototype, BasicArgs, OutputArgs, WCKeyable


@dataclass(frozen=True)
class EMHArguments(ArgumentsPrototype):
    """Arguments for :class:`~qurry.qurrent.hadamard_test.experiment.EMHExperiment`."""

    exp_name: str
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    degree: tuple[int, int]
    """The degree range."""

    @classmethod
    def ingest(cls, raw_dict: dict[str, Any]):
        """Ingest from a serialized dictionary.

        Args:
            raw_dict (dict[str, Any]): The raw read dictionary.
        """

        return cls(
            exp_name=raw_dict["exp_name"],
            degree=tuple(raw_dict["degree"]),
        )


class EMHMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Union[QuantumCircuit, WCKeyable]
    """The key or the circuit to execute."""
    degree: Union[int, tuple[int, int], None]
    """The degree range."""


class EMHOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard.output`."""

    degree: Union[int, tuple[int, int], None]
    """The degree range."""


SHORT_NAME = "qurrent_hadamard"
"""The short name of :class:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard`."""

ACRONYM = "EMH"
"""The abbreviation of :class:`~qurry.qurrent.hadamard_test.qurry.EntropyMeasureHadamard`."""
