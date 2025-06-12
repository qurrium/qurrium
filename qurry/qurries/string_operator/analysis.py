"""StringOperator - Analysis
(:mod:`qurry.qurries.string_operator.analysis`)

"""

from typing import Union, Optional, NamedTuple, Iterable, Type
import numpy as np

from .utils import AvailableStringOperatorTypes
from ...qurrium.analysis import AnalysisPrototype


class SOAnalysisInput(NamedTuple):
    """To set the analysis."""

    num_qubits: int
    """The number of qubits."""
    i: int
    """The index of beginning qubits in the quantum circuit."""
    k: int
    """The index of ending qubits in the quantum circuit."""
    str_op: AvailableStringOperatorTypes
    """The string operator."""
    shots: int
    """The number of shots."""


class SOAnalysisContent(NamedTuple):
    """The content of the analysis."""

    order: Optional[Union[float, np.float64]] = None
    """The order of the string operator."""


class StringOperatorAnalysis(AnalysisPrototype[SOAnalysisInput, SOAnalysisContent]):
    """The container for the analysis of :cls:`StringOperatorExperiment`."""

    __name__ = "SOAnalysis"

    @classmethod
    def input_type(cls) -> Type[SOAnalysisInput]:
        """The input instance type."""
        return SOAnalysisInput

    @classmethod
    def content_type(cls) -> Type[SOAnalysisContent]:
        """The content instance type."""
        return SOAnalysisContent

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []
