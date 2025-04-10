"""StringOperator - Analysis
(:mod:`qurry.qurries.string_operator.analysis`)

"""

from typing import Union, Optional, NamedTuple, Iterable
import numpy as np

from .utils import AvailableStringOperatorTypes
from ...qurrium.analysis import AnalysisPrototype


class StringOperatorAnalysis(AnalysisPrototype):
    """The container for the analysis of :cls:`StringOperatorExperiment`."""

    __name__ = "StringOperatorAnalysis"

    class AnalysisInput(NamedTuple):
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

    input: AnalysisInput

    class AnalysisContent(NamedTuple):
        """The content of the analysis."""

        order: Optional[Union[float, np.float64]] = None
        """The order of the string operator."""

        def __repr__(self):
            return f"AnalysisContent(order={self.order})"

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return []

    content: AnalysisContent
