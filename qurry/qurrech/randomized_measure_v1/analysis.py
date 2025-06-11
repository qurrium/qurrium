"""EchoListenRandomizedV1 - Analysis
(:mod:`qurry.qurrech.randomized_measure_v1.analysis`)

"""

from typing import Optional, NamedTuple, Iterable, Type

from ...qurrium.analysis import AnalysisPrototype


class ELRV1AnalysisInput(NamedTuple):
    """To set the analysis."""

    degree: tuple[int, int]
    """The range of partition."""
    shots: int
    """The number of shots."""
    unitary_loc: Optional[tuple[int, int]] = None
    """The location of the random unitary operator."""


class ELRV1AnalysisContent(NamedTuple):
    """The content of the analysis."""

    echo: float
    """The purity of the system."""
    echoSD: float
    """The standard deviation of the purity of the system."""
    echoCells: dict[int, float]
    """The echo of each cell of the system."""
    bitStringRange: tuple[int, int]
    """The qubit range of the subsystem."""

    measureActually: Optional[tuple[int, int]] = None
    """The qubit range of the measurement actually used."""
    countsNum: Optional[int] = None
    """The number of counts of the experiment."""
    takingTime: Optional[float] = None
    """The taking time of the selected system."""
    counts_used: Optional[Iterable[int]] = None
    """The index of the counts used.
    If not specified, then use all counts."""

    def __repr__(self):
        return f"ELRV1AnalysisContent(echo={self.echo}, and others)"


class EchoListenRandomizedV1Analysis(AnalysisPrototype[ELRV1AnalysisInput, ELRV1AnalysisContent]):
    """The analysis of loschmidt echo."""

    __name__ = "ELRV1Analysis"

    @classmethod
    def input_type(cls) -> Type[ELRV1AnalysisInput]:
        """The input instance type."""
        return ELRV1AnalysisInput

    @classmethod
    def content_type(cls) -> Type[ELRV1AnalysisContent]:
        """The content instance type."""
        return ELRV1AnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return [
            "echoCells",
        ]
