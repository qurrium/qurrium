"""Classical Shadow - Classical Shadow - Container
(:mod:`qurry.process.classical_shadow.container`)

"""

from typing import Union, TypedDict
import numpy as np


class ClassicalShadowBasic(TypedDict):
    """The basic information of the classical shadow."""

    average_classical_snapshots_rho: dict[int, np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    """The dictionary of average classical snapshots, 
    which uses the notation rho in 
    [Predicting many properties of a quantum system from very few measurements](
        https://doi.org/10.1038/s41567-020-0932-7).

    The numpy.array shape is (2, 2).
    """
    classical_registers_actually: list[int]
    """The list of the selected_classical_registers."""
    taking_time: float
    """The time taken for the calculation."""


class ClassicalShadowMeanRho(ClassicalShadowBasic):
    """The esitimations of the classical shadow from classical snapshots.

    Here, we use the notations that use in the supplementary material of
    [Predicting many properties of a quantum system from very few measurements](
        https://doi.org/10.1038/s41567-020-0932-7),

    """

    mean_of_rho: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    """The mean of single classical snapshots."""


class ClassicalShadowEstimation(ClassicalShadowBasic):
    """The esitimations of the classical shadow from classical snapshots.

    Here, we use the notations that use in the supplementary material of
    [Predicting many properties of a quantum system from very few measurements](
        https://doi.org/10.1038/s41567-020-0932-7),

    """

    estimate_of_given_operators: list[np.complex128]
    r"""The esitmation values of measurement primitive :math:`\mathcal{U}`."""
    corresponding_rhos: list[np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    r"""The corresponding rho of measurement primitive :math:`\mathcal{U}`."""
    # The accuracy of estimation
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
    r"""The maximum shadow norm, which is defined in the supplementary material.
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
    The :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` is maximum shadow norm.

    Due to its calculation is complex, we curently use the value of np.nan
    to represent the maximum shadow norm.
    """
    epsilon_upperbound: float
    r"""The upper bound of the prediction of accuracy, 
    which used the notation :math:`\epsilon`
    and mentioned in Theorem S1 in the supplementary material,
    the equation (S13) in the supplementary material.

    .. math::
        || O ||_{\text{shadow}}^2 \leq 4^n || O ||_{\infty}^2
    where :math:`O` is the any operator, and :math:`n` is the number of qubits,

    So we set the shadow norm as follows,
    .. math::
        \chi = || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}} \\
        \chi_{\infty} = 4^n || O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\infty}^2 \\
        \chi^2 \leq \chi_{\infty}
    and we can simplify the equation to:
    .. math::
        N = \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} \chi^2 
            \leq \frac{34}{\epsilon^2} \max_{1 \leq i \leq M} \chi_{\infty}^2
    Then get:
    .. math::
        \epsilon \leq \sqrt{\frac{34}{N}} \max_{1 \leq i \leq M} \chi_\infty
    """
    shadow_norm_upperbound: float
    r"""The largest shadow norm upper bound is defined as follows,
    .. math::
        || O ||_{\text{shadow}}^2 \leq 4^n || O ||_{\infty}^2
    where :math:`O` is the operator, and :math:`n` is the number of qubits,
    which mentioned in the paper at Theorem 1 (informal version).

    This is the worst scenario of the shadow norm
    for its scaling can be reduced to :math:`3^n || O ||_{\infty}^2`,
    which is the significantly lower bound than the worst case scenario.
    """


class ClassicalShadowPurity(ClassicalShadowBasic):
    """The expectation value of Rho."""

    purity: Union[float, np.float64]
    """The purity calculated by classical shadow."""
    entropy: Union[float, np.float64]
    """The entropy calculated by classical shadow."""


class ClassicalShadowComplex(
    ClassicalShadowEstimation, ClassicalShadowMeanRho, ClassicalShadowPurity
):
    """The expectation value of Rho and the purity calculated by classical shadow."""
