"""StringOperator - Analysis (:mod:`qurry.qurries.string_operator.analysis`)"""

from typing import Union, NamedTuple, Iterable, Type
import numpy as np

from .utils import StringOperatorLibType, StringOperatorDirection
from ...qurrium.analysis import AnalysisPrototype


class SOAnalysisInput(NamedTuple):
    """To set the analysis."""


class SOAnalysisContent(NamedTuple):
    """The content of the analysis."""

    order: Union[float, np.float64]
    """The order of the string operator."""
    num_qubits: int
    """The number of qubits."""
    shots: int
    """The number of shots."""
    i: int
    """The index of beginning qubits in the quantum circuit."""
    k: int
    """The index of ending qubits in the quantum circuit."""
    length: int
    """The length of the string operator, which is k - i + 1."""
    str_op: StringOperatorLibType = "i"
    """The string operator."""
    on_dir: StringOperatorDirection = "x"
    """The direction of the string operator, either 'x' or 'y'."""


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
