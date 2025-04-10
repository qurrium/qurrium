"""MagnetSquare - Analysis
(:mod:`qurry.qurries.magnet_square.analysis`)

"""

from typing import Union, Optional, NamedTuple, Iterable
import numpy as np

from ...qurrium.analysis import AnalysisPrototype


class MagnetSquareAnalysis(AnalysisPrototype):
    """The container for the analysis of :cls:`MagnetSquareExperiment`."""

    __name__ = "MagnetSquareAnalysis"

    class AnalysisInput(NamedTuple):
        """To set the analysis."""

        num_qubits: int
        """The number of qubits."""
        shots: int
        """The number of shots."""

    input: AnalysisInput

    class AnalysisContent(NamedTuple):
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
            return f"AnalysisContent(magnet_square={self.magnet_square}, and others)"

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []

    content: AnalysisContent
