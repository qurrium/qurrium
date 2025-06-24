"""EchoListenHadamard - Analysis
(:mod:`qurry.qurrech.hadamard_test.analysis`)

"""

from typing import NamedTuple, Iterable, Type

from ...qurrium.analysis import AnalysisPrototype


class ELHAnalysisInput(NamedTuple):
    """To set the analysis."""


class ELHAnalysisContent(NamedTuple):
    """The content of the analysis."""

    echo: float
    """The purity of the system."""


class EchoListenHadamardAnalysis(AnalysisPrototype[ELHAnalysisInput, ELHAnalysisContent]):
    """The analysis for calculating entangled entropy with more information combined."""

    __name__ = "ELHAnalysis"

    @classmethod
    def input_type(cls) -> Type[ELHAnalysisInput]:
        """The input instance type."""
        return ELHAnalysisInput

    @classmethod
    def content_type(cls) -> Type[ELHAnalysisContent]:
        """The content instance type."""
        return ELHAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
