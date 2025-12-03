"""EchoListenHadamard - Analysis (:mod:`qurry.qurrech.hadamard_test.analysis`)"""

from typing import Optional, Any
from dataclasses import dataclass

from .arguments import ELHArguments
from ...qurrium import (
    Commonparams,
    AnalysisPrototype,
    AnalyzeArgs,
    AnalysisMiddlewarePrototype,
    ProcessEntriesPrototype,
    AnalysisResultsPrototype,
)
from ...process.hadamard_test import hadamard_overlap_echo


class ELHAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
    and :meth:`~qurry.qurrech.hadamard_test.experiment.ELHExperiment.analyze`.

    The post-processing of Hadamard test does not need any input.
    """


@dataclass(frozen=True)
class ELHAnalysisMiddleware(AnalysisMiddlewarePrototype):
    """The middleware entries between analyze and actual post-processing function."""

    __name__ = "ELHAnalysisMiddleware"


@dataclass(frozen=True)
class ELHProcessEntries(ProcessEntriesPrototype):
    """The entries for post-processing."""

    __name__ = "ELHProcessEntries"


@dataclass(frozen=True)
class ELHDefaultResults(AnalysisResultsPrototype):
    """The default results of :class:`~qurry.qurrech.hadamard_test.analysis.ELHAnalysis`,
    which contains only wavefunction overlap or Loschmidt echo."""

    echo: float
    """The wavefunction overlap or Loschmidt echo of the system."""

    __name__ = "ELHDefaultResults"


class ELHAnalysis(
    AnalysisPrototype[
        ELHArguments,
        ELHAnalyzeArgs,
        ELHAnalysisMiddleware,
        ELHProcessEntries,
        ELHDefaultResults,
    ]
):
    """The instance for the analysis of
    :class:`~qurry.qurrech.hadamard_test.experiment.ELHExperiment`.
    """

    __name__ = "ELHAnalysis"

    @classmethod
    def middleware_entries_type(cls) -> type[ELHAnalysisMiddleware]:
        """The middleware entries type for this analysis."""
        return ELHAnalysisMiddleware

    @classmethod
    def postprocess_entries_type(cls) -> type[ELHProcessEntries]:
        """The post-processing entries type for this analysis."""
        return ELHProcessEntries

    @classmethod
    def available_results_types(cls):
        """The results type for this analysis."""
        return {"default": ELHDefaultResults}

    @classmethod
    def quantities(cls, shots: int, counts: list[dict[str, int]]):
        """Calculate wavefunction overlap with more information combined.

        Args:
            shots (int): Shots of the experiment on quantum machine.
            counts (list[dict[str, int]]): Counts of the experiment on quantum machine.

        Returns:
            A dictionary contains wavefunction overlap or Loschmidt echo.
        """

        return hadamard_overlap_echo(shots=shots, counts=counts)

    @classmethod
    def generate_entries(
        cls,
        arguments: ELHArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: ELHAnalyzeArgs,
    ) -> tuple[ELHAnalyzeArgs, ELHAnalysisMiddleware, ELHProcessEntries]:
        """Generate the entries for analysis.

        Hint:
            Hadamard test does not need any specific entries.

        Args:
            arguments (EMHArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (EMHAnalyzeArgs): The analyze arguments.

        Returns:
            The generated entries for analysis.
        """
        middleware_entries = ELHAnalysisMiddleware()
        postprocess_entries = ELHProcessEntries()

        return analyze_arguments, middleware_entries, postprocess_entries

    @classmethod
    def perform_analysis(
        cls,
        arguments: ELHArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: ELHAnalyzeArgs,
        serial: int,
        outfields: Optional[dict[str, Any]] = None,
        datetime: Optional[str] = None,
    ):
        """Perform the analysis for the experiment.

        Args:
            arguments (ELHArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (ELHAnalyzeArgs): The analyze arguments.
            serial (int): The serial number of the analysis.
            outfields (Optional[dict[str, Any]], optional):
                The unused arguments of the analysis. Defaults to None.
            datetime (Optional[str], optional):
                The datetime of the analysis. Defaults to None.

        Returns:
            AnalysisPrototype: The result of the analysis.
        """

        analyze_arguments, middleware_entries, postprocess_entries = cls.generate_entries(
            arguments, commonparams, counts, analyze_arguments
        )
        hadamard_results_dict = cls.quantities(shots=commonparams.shots, counts=counts)
        results = ELHDefaultResults(echo=float(hadamard_results_dict["echo"]))

        return cls(
            analyze_arguments=analyze_arguments,
            middleware_entries=middleware_entries,
            postprocess_entries=postprocess_entries,
            results={"default": results},
            serial=serial,
            outfields=outfields,
            datetime=datetime,
        )
