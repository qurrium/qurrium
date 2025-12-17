"""SamplingExecuter - Arguments (:mod:`qurry.qurries.samplingqurry.arguments`)"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class QurryArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurries.samplingqurry.experiment.QurryExperiment`."""

    sampling: int = 1
    """The number of sampling."""


class QurryMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurries.samplingqurry.qurry.QurryV9.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    sampling: int
    """The number of sampling."""


class QurryOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurries.samplingqurry.qurry.QurryV9.output`."""

    sampling: int
    """The number of sampling."""


class QurryAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis` and
    :meth:`~qurry.qurries.samplingqurry.experiment.QurryExperiment.analyze`.
    """


SHORT_NAME = "sampling_executer"
"""The short name of
:class:`~qurry.qurries.samplingqurry.experiment.QurryExperiment`"""
