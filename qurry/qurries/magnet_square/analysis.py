"""MagnetSquare - Analysis
(:mod:`qurry.qurries.magnet_square.analysis`)

"""

from typing import Union, Optional, NamedTuple, Iterable, Type
import numpy as np

from ...qurrium.analysis import AnalysisPrototype


class MagnetSquareAnalysisInput(NamedTuple):
    """To set the analysis."""

    num_qubits: int
    """The number of qubits."""
    shots: int
    """The number of shots."""


class MagnetSquareAnalysisContent(NamedTuple):
    """The content of the analysis."""

    magnet_square: Optional[Union[float, np.float64]] = None
    """Magnetic Square."""
    magnet_square_cells: Optional[dict[int, Union[float, np.float64]]] = None
    """Magnetic Square cells."""
    counts_num: Optional[int] = None
    """Number of counts."""
    taking_time: Optional[float] = None
    """Taking time."""

    def __repr__(self):
        return f"MagnetSquareAnalysisContent(magnet_square={self.magnet_square}, and others)"


class MagnetSquareAnalysis(
    AnalysisPrototype[MagnetSquareAnalysisInput, MagnetSquareAnalysisContent]
):
    """The container for the analysis of :cls:`MagnetSquareExperiment`."""

    __name__ = "MagnetSquareAnalysis"

    @classmethod
    def input_type(cls) -> Type[MagnetSquareAnalysisInput]:
        """The input instance type."""
        return MagnetSquareAnalysisInput

    @classmethod
    def content_type(cls) -> Type[MagnetSquareAnalysisContent]:
        """The content instance type."""
        return MagnetSquareAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
