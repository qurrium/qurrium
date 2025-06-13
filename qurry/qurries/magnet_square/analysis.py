"""MagnetSquare - Analysis (:mod:`qurry.qurries.magnet_square.analysis`)"""

from typing import Union, Optional, NamedTuple, Iterable, Type
import numpy as np
import numpy.typing as npt

from ...qurrium.analysis import AnalysisPrototype


class MSAnalysisInput(NamedTuple):
    """To set the analysis."""


class MSAnalysisContent(NamedTuple):
    """The content of the analysis."""

    magnet_square: Union[float, np.float64]
    """Magnetic Square."""
    num_qubits: int
    """The number of qubits."""
    shots: int
    """The number of shots."""
    unitary_operator: Union[str, npt.NDArray[np.float64], npt.NDArray[np.complex128]]
    """The numpy array of the unitary operator or a string representing the axis of rotation."""
    magnet_square_cells: Union[dict[int, np.float64], dict[int, np.float64]]
    """Magnetic Square cells."""
    taking_time: Optional[float] = None
    """Taking time."""

    def __repr__(self):
        return (
            "MSAnalysisContent("
            f"magnet_square={self.magnet_square}, "
            f"unitary_operator={self.unitary_operator}, "
            "and others)"
        )


class MagnetSquareAnalysis(AnalysisPrototype[MSAnalysisInput, MSAnalysisContent]):
    """The container for the analysis of :cls:`MagnetSquareExperiment`."""

    __name__ = "MSAnalysis"

    @classmethod
    def input_type(cls) -> Type[MSAnalysisInput]:
        """The input instance type."""
        return MSAnalysisInput

    @classmethod
    def content_type(cls) -> Type[MSAnalysisContent]:
        """The content instance type."""
        return MSAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
