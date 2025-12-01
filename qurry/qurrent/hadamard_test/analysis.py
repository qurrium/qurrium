"""EntropyMeasureHadamard - Analysis (:mod:`qurry.qurrent.hadamard_test.analysis`)"""

from typing import Optional, Any
from dataclasses import dataclass

from .arguments import EMHArguments
from ...qurrium import (
    Commonparams,
    AnalysisPrototype,
    AnalyzeArgs,
    AnalyzeEntriesPrototype,
    AnalyzeResultsPrototype,
)
from ...process.hadamard_test import hadamard_entangled_entropy


class EMHAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis`.
    and :meth:`~qurry.qurrent.hadamard_test.experiment.EntropyMeasureHadamard.analyze`.

    The post-processing of Hadamard test does not need any input.
    """


@dataclass(frozen=True)
class EMHAnalyzeMiddlewareEntries(AnalyzeEntriesPrototype):
    """To set the analysis."""


@dataclass(frozen=True)
class EMHAnalyzePostProcessingEntries(AnalyzeEntriesPrototype):
    """The input entries for post-processing."""


@dataclass(frozen=True)
class EMHAnalyzeResults(AnalyzeResultsPrototype):
    """The content of the analysis."""

    purity: float
    """The purity of the system."""
    entropy: float
    """The entanglement entropy of the system."""

    __name__ = "EntropyMeasureHadamardAnalyzeResults"

    def side_product_fields(self) -> tuple[str, ...]:
        """The fields that will be stored as side product.

        Hint:
            In Hadamard test, all fields are main results.
        """
        return ()


class EMHAnalysis(
    AnalysisPrototype[
        EMHArguments,
        EMHAnalyzeArgs,
        EMHAnalyzeMiddlewareEntries,
        EMHAnalyzePostProcessingEntries,
        EMHAnalyzeResults,
    ]
):
    """The instance for the analysis of
    :class:`~qurry.qurrent.hadamard_test.experiment.EntropyMeasureHadamardExperiment`.
    """

    __name__ = "EMHAnalysis"

    @classmethod
    def middleware_entries_type(cls) -> type[EMHAnalyzeMiddlewareEntries]:
        """The middleware entries type for this analysis."""
        return EMHAnalyzeMiddlewareEntries

    @classmethod
    def postprocess_entries_type(cls) -> type[EMHAnalyzePostProcessingEntries]:
        """The post-processing entries type for this analysis."""
        return EMHAnalyzePostProcessingEntries

    @classmethod
    def results_type(cls) -> dict[str, type[AnalyzeResultsPrototype]]:
        """The results type for this analysis."""
        return {"default": EMHAnalyzeResults}

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
    ) -> tuple[EMHAnalyzeArgs, EMHAnalyzeMiddlewareEntries, EMHAnalyzePostProcessingEntries]:
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
        middleware_entries = EMHAnalyzeMiddlewareEntries()
        postprocess_entries = EMHAnalyzePostProcessingEntries()

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
        results = EMHAnalyzeResults(
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
