"""WavesExecuter - Analysis (:mod:`qurry.qurrium.wavesqurry.analysis`)

It is only for pendings and retrieve to remote backend.
"""

from typing import NamedTuple, Iterable

from ...qurrium import AnalysisPrototype


class WEAnalysisInput(NamedTuple):
    """To set the analysis."""

    ultimate_question: str
    """ULtImAte QueStIoN."""


class WEAnalysisContent(NamedTuple):
    """Analysis content."""

    ultimate_answer: int
    """~The Answer to the Ultimate Question of Life, The Universe, and Everything.~"""
    dummy: int
    """Just a dummy field."""


class WavesExecuterAnalysis(AnalysisPrototype[WEAnalysisInput, WEAnalysisContent]):
    """The analysis of the experiment."""

    __name__ = "WavesQurryAnalysis"

    @classmethod
    def input_type(cls):
        """The input instance type."""
        return WEAnalysisInput

    @classmethod
    def content_type(cls):
        """The content instance type."""
        return WEAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
