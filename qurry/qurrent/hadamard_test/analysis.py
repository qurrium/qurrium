"""EntropyMeasureHadamard - Analysis (:mod:`qurry.qurrent.hadamard_test.analysis`)"""

from typing import Optional, Any
from dataclasses import dataclass

from .arguments import EMHArguments
from ...qurrium import (
    Commonparams,
    AnalysisPrototype,
    AnalyzeArgs,
    AnalysisMiddlewarePrototype,
    ProcessEntriesPrototype,
    AnalysisResultsPrototype,
)
from ...process.hadamard_test import hadamard_entangled_entropy


class EMHAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
    and :meth:`~qurry.qurrent.hadamard_test.experiment.EntropyMeasureHadamard.analyze`.

    The post-processing of Hadamard test does not need any input.
    """


@dataclass(frozen=True)
class EMHAnalysisMiddleware(AnalysisMiddlewarePrototype):
    """The middleware entries between analyze and actual post-processing function."""

    __name__ = "EMHAnalysisMiddleware"


@dataclass(frozen=True)
class EMHProcessEntries(ProcessEntriesPrototype):
    """The entries for post-processing."""

    __name__ = "EMHProcessEntries"


@dataclass(frozen=True)
class EMHDefaultResults(AnalysisResultsPrototype):
    """The default results of :cls:`~qurry.qurrent.hadamard_test.analysis.EMHAnalysis`,
    which contains only purity and entanglement entropy."""

    purity: float
    """The purity of the system."""
    entropy: float
    """The entanglement entropy of the system."""

    __name__ = "EMHDefaultResults"


class EMHAnalysis(
    AnalysisPrototype[
        EMHArguments,
        EMHAnalyzeArgs,
        EMHAnalysisMiddleware,
        EMHProcessEntries,
        EMHDefaultResults,
    ]
):
    """The instance for the analysis of
    :class:`~qurry.qurrent.hadamard_test.experiment.EntropyMeasureHadamardExperiment`.
    """

    __name__ = "EMHAnalysis"

    @classmethod
    def middleware_entries_type(cls) -> type[EMHAnalysisMiddleware]:
        """The middleware entries type for this analysis."""
        return EMHAnalysisMiddleware

    @classmethod
    def postprocess_entries_type(cls) -> type[EMHProcessEntries]:
        """The post-processing entries type for this analysis."""
        return EMHProcessEntries

    @classmethod
    def results_type(cls) -> dict[str, type[AnalysisResultsPrototype]]:
        """The results type for this analysis."""
        return {"default": EMHDefaultResults}

    @classmethod
    def quantities(cls, shots: int, counts: list[dict[str, int]]):
        """Calculate entangled entropy with more information combined.

        Args:
            shots (int): Shots of the experiment on quantum machine.
            counts (list[dict[str, int]]): Counts of the experiment on quantum machine.

        Returns:
            dict[str, float]: A dictionary contains
                purity, entropy.
        """

        return hadamard_entangled_entropy(shots=shots, counts=counts)

    @classmethod
    def generate_entries(
        cls,
        arguments: EMHArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: EMHAnalyzeArgs,
    ) -> tuple[EMHAnalyzeArgs, EMHAnalysisMiddleware, EMHProcessEntries]:
        """Generate the entries for analysis.

        Hint:
            Hadamard test does not need any specific entries.

        Args:
            arguments (EMHArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (EMHAnalyzeArgs): The analyze arguments.

        Returns:
            tuple[
                EMHAnalyzeArgs,
                EMHAnalyzeMiddlewareEntries,
                EMHAnalyzePostProcessingEntries,
            ]: The generated entries for analysis.
        """
        middleware_entries = EMHAnalysisMiddleware()
        postprocess_entries = EMHProcessEntries()

        return analyze_arguments, middleware_entries, postprocess_entries

    @classmethod
    def perform_analysis(
        cls,
        arguments: EMHArguments,
        commonparams: Commonparams,
        counts: list[dict[str, int]],
        analyze_arguments: EMHAnalyzeArgs,
        serial: int,
        outfields: Optional[dict[str, Any]] = None,
        datetime: Optional[str] = None,
    ):
        """Perform the analysis for the experiment.

        Args:
            arguments (EMHArguments): The arguments for the experiment.
            commonparams (Commonparams): The common parameters for the experiment.
            counts (list[dict[str, int]]): The counts from the experiment.
            analyze_arguments (EMHAnalyzeArgs): The analyze arguments.
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
        hadamard_results_dict = cls.quantities(
            shots=commonparams.shots,
            counts=counts,
        )
        results = EMHDefaultResults(
            purity=float(hadamard_results_dict["purity"]),
            entropy=float(hadamard_results_dict["entropy"]),
        )

        return cls(
            analyze_arguments=analyze_arguments,
            middleware_entries=middleware_entries,
            postprocess_entries=postprocess_entries,
            results={"default": results},
            serial=serial,
            outfields=outfields,
            datetime=datetime,
        )
