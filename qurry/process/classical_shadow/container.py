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

    estimate_of_given_operators: list[np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    r"""The result of measurement primitive :math:`\mathcal{U}`."""
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
