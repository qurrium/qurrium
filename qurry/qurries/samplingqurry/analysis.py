"""SamplingExecuter - Analysis (:mod:`qurry.qurrium.samplingqurry.analysis`)

It is only for pendings and retrieve to remote backend.
"""

from typing import NamedTuple, Iterable

from ...qurrium import AnalysisPrototype


class QurryAnalysisInput(NamedTuple):
    """To set the analysis."""

    ultimate_question: str
    """ULtImAte QueStIoN."""


class QurryAnalysisContent(NamedTuple):
    """Analysis content."""

    ultimate_answer: int
    """~The Answer to the Ultimate Question of Life, The Universe, and Everything.~"""
    dummy: int
    """Just a dummy field."""


class QurryAnalysis(AnalysisPrototype[QurryAnalysisInput, QurryAnalysisContent]):
    """Example of QurryAnalysis."""

    __name__ = "QurryAnalysis"

    @classmethod
    def input_type(cls):
        """The input instance type."""
        return QurryAnalysisInput

    @classmethod
    def content_type(cls):
        """The content instance type."""
        return QurryAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
