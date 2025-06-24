"""ZDirMagnetSquare - Analysis (:mod:`qurry.qurries.magnet_square_z.analysis`)"""

from typing import Union, Optional, NamedTuple, Iterable, Type
import numpy as np

from ...qurrium.analysis import AnalysisPrototype


class ZDirMSAnalysisInput(NamedTuple):
    """To set the analysis."""


class ZDirMSAnalysisContent(NamedTuple):
    """The content of the analysis."""

    magnet_square: Union[float, np.float64]
    """Magnetic Square."""
    num_qubits: int
    """The number of qubits."""
    shots: int
    """The number of shots."""
    magnet_square_cells: Union[dict[int, np.float64], dict[int, np.float64]]
    """Magnetic Square cells."""
    taking_time: Optional[float] = None
    """Taking time."""

    def __repr__(self):
        return f"ZDirMSAnalysisContent(magnet_square={self.magnet_square}, and others)"


class ZDirMagnetSquareAnalysis(AnalysisPrototype[ZDirMSAnalysisInput, ZDirMSAnalysisContent]):
    """The container for the analysis of :cls:`ZDirMagnetSquareExperiment`."""

    __name__ = "ZDirMSAnalysis"

    @classmethod
    def input_type(cls) -> Type[ZDirMSAnalysisInput]:
        """The input instance type."""
        return ZDirMSAnalysisInput

    @classmethod
    def content_type(cls) -> Type[ZDirMSAnalysisContent]:
        """The content instance type."""
        return ZDirMSAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
