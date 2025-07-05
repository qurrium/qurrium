"""WavesExecuter - Arguments (:mod:`qurry.qurrium.wavesqurry.arguments`)

It is only for pendings and retrieve to remote backend.
"""

from typing import Optional, Union
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium import ArgumentsPrototype
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class WavesExecuterArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurrium.wavesqurry.experiment.WavesExecuterExperiment`."""


class WavesExecuterMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrium.wavesqurry.qurry.WavesExecuter.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    waves: Optional[list[Union[QuantumCircuit, Hashable]]]


class WavesExecuterOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`qurry.qurrium.wavesqurry.qurry.WavesExecuter.output`."""


class WavesExecuterAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis` and
    :meth:`~qurry.qurrium.wavesqurry.experiment.WavesExecuterExperiment.analyze`.
    """


SHORT_NAME = "waves_executer"
"""The short name of the experiment. """
