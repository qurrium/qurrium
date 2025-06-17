"""ShadowUnveil - Analysis
(:mod:`qurry.qurrent.classical_shadow.analysis`)

"""

from typing import Optional, NamedTuple, Iterable, Any, Type
import numpy as np

from ...qurrium.analysis import AnalysisPrototype


class SUAnalysisInput(NamedTuple):
    """To set the analysis."""

    shots: int
    """The number of shots."""
    num_qubits: int
    """The number of qubits."""
    selected_qubits: list[int]
    """The selected qubits."""
    registers_mapping: dict[int, int]
    """The mapping of the classical registers with quantum registers.

    .. code-block:: python
        {
            0: 0, # The quantum register 0 is mapped to the classical register 0.
            1: 1, # The quantum register 1 is mapped to the classical register 1.
            5: 2, # The quantum register 5 is mapped to the classical register 2.
            7: 3, # The quantum register 7 is mapped to the classical register 3.
        }

    The key is the index of the quantum register with the numerical order.
    The value is the index of the classical register with the numerical order.
    """
    bitstring_mapping: Optional[dict[int, int]]
    """The mapping of the bitstring with the classical registers.
    When there are mulitple classical registers, 
    the bitstring is the concatenation of the classical registers with space on bitstring.
    For example, there are three registers with the size of 4, 4, and 6, 
    which the first six bits are for the randomized measurement.

    .. code-block:: python
        {'010000 0100 0001': 1024}
        # The bitstring is '010000 0100 0001'.
        # The last four bits are the first classical register.
        # The middle four bits are the second classical register.
        # The first six bits are the last classical register for the randomized measurement.

    So, the mapping will be like this.

    .. code-block:: python

        {
            0: 10, # The classical register 0 is mapped to the bitstring on the index 0.
            1: 11, # The classical register 0 is mapped to the bitstring on the index 1.
            2: 12, # The classical register 0 is mapped to the bitstring on the index 2.
            3: 13, # The classical register 0 is mapped to the bitstring on the index 3.
            4: 14, # The classical register 0 is mapped to the bitstring on the index 4.
            5: 15, # The classical register 0 is mapped to the bitstring on the index 5.
        }

    But, if there is only one classical register, 
    the bitstring will map to the classical register directly.

    .. code-block:: python
        {'010000': 1024}

    Will be like this.

    .. code-block:: python
        {
            0: 0, # The classical register 0 is mapped to the bitstring on the index 0.
            1: 1, # The classical register 0 is mapped to the bitstring on the index 1.
            2: 2, # The classical register 0 is mapped to the bitstring on the index 2.
            3: 3, # The classical register 0 is mapped to the bitstring on the index 3.
            4: 4, # The classical register 0 is mapped to the bitstring on the index 4.
            5: 5, # The classical register 0 is mapped to the bitstring on the index 5.
        }

    """
    unitary_located: Optional[list[int]] = None
    """The range of the unitary operator."""


class SUAnalysisContent(NamedTuple):
    """The content of the analysis."""

    average_classical_snapshots_rho: dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    """The dictionary of Rho M."""
    classical_registers_actually: list[int]
    """The list of the selected_classical_registers."""
    taking_time: float
    """The time taken for the calculation."""
    # The mean of Rho
    mean_of_rho: np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    """The expectation value of Rho."""
    # The trace of Rho square
    purity: float
    """The purity calculated by classical shadow."""
    entropy: float
    """The entropy calculated by classical shadow."""
    # esitimation of given operators
    estimate_of_given_operators: list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    r"""The result of measurement primitive :math:`\mathcal{U}`."""
    corresponding_rhos: list[np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    r"""The corresponding rho of measurement primitive :math:`\mathcal{U}`."""
    accuracy_prob_comp_delta: float
    r"""The probabiltiy complement of accuracy, which used the notation :math:`\delta`
    and mentioned in Theorem S1 in the supplementary material,
    the equation (S13) in the supplementary material.
    The probabiltiy of accuracy is :math:`1 - \delta`.

    The number of given operators and the accuracy parameters will 
    be used to decide the number of estimators K 
    from the equation (S13) in the supplementary material.

    .. math::
        K = 2 \log(2M / \delta)

    where :math:`\delta` is the probabiltiy complement of accuracy,
    and :math:`M` is the number of given operators.

    But we can see :math:`K` will be not the integer value of the result of the equation.
    So, we will use the ceil value of the result of the equation.
    And recalculate the probabiltiy complement of accuracy from this new value of :math:`K`.
    """
    num_of_estimators_k: int
    r"""The number of esitmators, which used the notation K
    and mentioned in Algorithm 1 in the paper,
    Theorem S1 in the supplementary material,
    the equation (S13) in the supplementary material.

    We can calculate the number of esitmator K from the equation (S13) 
    in the supplementary material, the equation (S13) is as follows,
    .. math::
        K = 2 \log(2M / \delta)

    where :math:`\delta` is the probabiltiy complement of accuracy,
    and :math:`M` is the number of given operators.

    But we can see :math:`K` will be not the integer value of the result of the equation.
    So, we will use the ceil value of the result of the equation.
    And recalculate the probabiltiy complement of accuracy from this new value of :math:`K`.
    """

    accuracy_predict_epsilon: float
    r"""The prediction of accuracy, which used the notation :math:`\epsilon`
    and mentioned in Theorem S1 in the supplementary material,
    the equation (S13) in the supplementary material.

    We can calculate the prediction of accuracy :math:`\epsilon` from the equation (S13)
    in the supplementary material, the equation (S13) is as follows,
    .. math::
        N = \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} 
        || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2

    where :math:`\epsilon` is the prediction of accuracy,
    and :math:`M` is the number of given operatorsm
    and :math:`N` is the number of classical snapshots.
    The :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` is maximum shadow norm,
    which is defined in the supplementary material with value between 0 and 1.
    """
    maximum_shadow_norm: float
    r"""The maximum shadow norm, which is defined in the supplementary material 
    with value between 0 and 1.
    The maximum shadow norm is used to calculate the prediction of accuracy :math:`\epsilon`
    from the equation (S13) in the supplementary material.

    We can calculate the prediction of accuracy :math:`\epsilon` from the equation (S13)
    in the supplementary material, the equation (S13) is as follows,
    .. math::
        N = \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} 
        || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2

    where :math:`\epsilon` is the prediction of accuracy,
    and :math:`M` is the number of given operatorsm
    and :math:`N` is the number of classical snapshots.
    The :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` is maximum shadow norm,
    which is defined in the supplementary material with value between 0 and 1.

    Due to maximum shadow norm is complex and it is a norm,
    we suppose we have the worst case scenario,
    where the maximum shadow norm is 1 as default.
    Thus, we can simplify the equation to:
    .. math::
        N = \frac{34}{\epsilon^2}
    """

    def __repr__(self):
        return f"SUAnalysisContent(purity={self.purity}, entropy={self.entropy}, and others)"


FIELDS_REMAPPING = {
    "rho_m_dict": "average_classical_snapshots_rho",
    "expect_rho": "mean_of_rho",
}
NEW_FIELDS_DEFAULTS = {
    "average_classical_snapshots_rho": {},
    "mean_of_rho": np.zeros((1, 1), dtype=np.complex128),
    "classical_registers_actually": [],
    "taking_time": 0.0,
    "purity": np.nan,
    "entropy": np.nan,
    "estimate_of_given_operators": [],
    "corresponding_rhos": [],
    "accuracy_prob_comp_delta": np.nan,
    "num_of_estimators_k": 0,
    "accuracy_predict_epsilon": np.nan,
    "maximum_shadow_norm": np.nan,
}


class ShadowUnveilAnalysis(AnalysisPrototype[SUAnalysisInput, SUAnalysisContent]):
    """The container for the analysis of :cls:`EntropyRandomizedExperiment`."""

    __name__ = "SUAnalysis"

    @classmethod
    def input_type(cls) -> Type[SUAnalysisInput]:
        """The type of the input for the analysis."""
        return SUAnalysisInput

    @classmethod
    def content_type(cls) -> Type[SUAnalysisContent]:
        """The type of the content for the analysis."""
        return SUAnalysisContent

    @classmethod
    def deprecated_fields_converts(
        cls, main: dict[str, Any], side: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert deprecated fields to new fields.

        This method should be implemented in the subclass if there are deprecated fields
        that need to be converted.

        Args:
            main (dict[str, Any]): The main product dict.
            side (dict[str, Any]): The side product dict.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:
                The converted main and side product dicts.
        """

        if "expect_rho" not in main and "rho_m_dict" not in main:
            # If neither expect_rho nor rho_m_dict is present, return as is.
            return main, side

        main["mean_of_rho"] = main.pop("expect_rho")
        side["average_classical_snapshots_rho"] = side.pop("rho_m_dict")
        for k, v in NEW_FIELDS_DEFAULTS.items():
            if k not in main:
                main[k] = v
        return main, side

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return ["average_classical_snapshots_rho", "corresponding_rhos"]
