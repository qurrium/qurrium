"""EntropyMeasureHadamard - Analysis
(:mod:`qurry.qurrent.hadamard_test.analysis`)

"""

from typing import NamedTuple, Iterable, Type

from ...qurrium.analysis import AnalysisPrototype


class EMHAnalysisInput(NamedTuple):
    """To set the analysis."""


class EMHAnalysisContent(NamedTuple):
    """The content of the analysis."""

    purity: float
    """The purity of the system."""
    entropy: float
    """The entanglement entropy of the system."""

    def __repr__(self):
        return f"EMHAnalysisContent(purity={self.purity}, entropy={self.entropy})"


class EntropyMeasureHadamardAnalysis(AnalysisPrototype[EMHAnalysisInput, EMHAnalysisContent]):
    """The instance for the analysis of :cls:`EntropyHadamardExperiment`."""

    __name__ = "EMHAnalysis"

    @classmethod
    def input_type(cls) -> Type[EMHAnalysisInput]:
        """The input instance type."""
        return EMHAnalysisInput

    @classmethod
    def content_type(cls) -> Type[EMHAnalysisContent]:
        """The content instance type."""
        return EMHAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
