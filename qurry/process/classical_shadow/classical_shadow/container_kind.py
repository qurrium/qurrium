"""Post Processing - Classical Shadow -  Classical Shadow - Container/Purity Value Kind
(:mod:`qurry.process.classical_shadow.classical_shadow.container_kind`)
"""

from typing import Union, TypedDict, Literal
import numpy as np

from ..rho_process import RhoMethod, RhoMethodType
from ..all_trace_process import TraceMethod, TraceMethodType


PurityValueKind = Literal["multi_shots", "single_shots", "bitwise"]
"""The kind of purity value calculation.
This will depend on the rho_method and trace_method.

- "multi_shots":
    The *rho_method is one of the multi_shots methods* **and** *trace_method is one of the
    matrix operation methods*.

    .. code-block:: python

        (
            rho_method in [
                "multi_shots_proto", 
                "multi_shots", 
                "multi_shots_vectorized",
            ]
        ) and (
            trace_method in [
                "trace_of_matmul", 
                "einsum_ij_ji", 
                "quick_trace_of_matmul",
                "einsum_aij_bji_to_ab_numpy", 
                "einsum_aij_bji_to_ab_jax",
            ]
        )

- "single_shots":
    The *rho_method is one of the single_shots methods* **and** *trace_method is one of the
    matrix operation methods*, or the *trace_method is one of the non-matrix operation methods
    except "bitwise_py"*.
    
    .. code-block:: python

        (
            trace_method in [
                "nomatmul_trace_py", 
                "nomatmul_trace_rust",
            ]
        ) or (
            (
                rho_method in [
                    "single_shots_proto", 
                    "single_shots", 
                    "single_shots_vectorized",
                ]
            ) and (
                trace_method in [
                    "trace_of_matmul", 
                    "einsum_ij_ji", 
                    "quick_trace_of_matmul",
                    "einsum_aij_bji_to_ab_numpy", 
                    "einsum_aij_bji_to_ab_jax",
                ]
            )
        )

- "bitwise":
    The *trace_method is "bitwise_py"* no matter what the rho_method is.
    
    .. code-block:: python

        (trace_method in ["bitwise_py"])
"""


def purity_value_kind(rho_method: RhoMethodType, trace_method: TraceMethodType) -> PurityValueKind:
    """Get the kind of purity value calculation.

    Args:
        rho_method (RhoMethodType, optional):
            It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
            "single_shots_proto", "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots_proto": Use Numpy to calculate the rho_m.
            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots_proto": Use Numpy to calculate the rho_m
                with converted single shot counts.
            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        trace_method (TraceMethodType, optional):
            The method to calculate the trace of rho.

            - Matrix operation methods:
                - "trace_of_matmul": Use `np.trace(np.matmul(rho_m1, rho_m2))`
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_ij_ji": Use `np.einsum("ij,ji", rho_m1, rho_m2)`
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_aij_bji_to_ab_numpy": Use
                    `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is not available.
                - "einsum_aij_bji_to_ab_jax": Use
                    `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is available.

            For the matrix operation methods, it will require rho has been calculated first.

            - Non-matrix operation methods:
                - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
                - "nomatmul_trace_rust": Use Rust implementation via PyO3.
                - "bitwise_py": Use pure Python bitwise implementation.

            For the non-matrix operation methods, it will directly calculate the trace from
            the counts and random basis.

            The default method is "bitwise_py", which is the fastest option.

    Returns:
        PurityValueKind: The kind of purity value calculation.
    """
    if isinstance(rho_method, str):
        rho_method = RhoMethod.from_string(rho_method)
    if isinstance(trace_method, str):
        trace_method = TraceMethod.from_string(trace_method)

    if trace_method.is_bitwise_method():
        return "bitwise"
    if not trace_method.is_nomatop_method() and rho_method.is_multi_method():
        return "multi_shots"
    return "single_shots"


def default_method_on_value_kind(value_kind: PurityValueKind) -> tuple[RhoMethod, TraceMethod]:
    """Get the default method on each kind of purity value calculation.

    Args:
        purity_value_kind (PurityValueKind):
            The kind of purity value calculation.

    Raises:
        ValueError: If the purity value kind is not recognized.

    Returns:
        tuple[RhoMethod, TraceMethod]: The default (rho_method, trace_method).
    """
    if value_kind == "multi_shots":
        return RhoMethod.get_default(), TraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY
    if value_kind == "single_shots":
        return RhoMethod.get_default(), TraceMethod.NOMATMUL_TRACE_RUST
    if value_kind == "bitwise":
        return RhoMethod.get_default(), TraceMethod.BITWISE_PY
    raise ValueError(f"Unknown purity value kind: {value_kind}")


class ClassicalShadowBasic(TypedDict):
    """The basic information of the classical shadow."""

    average_classical_snapshots_rho: dict[int, np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    """The dictionary of average classical snapshots, 
    which uses the notation rho in 
    `Predicting many properties of a quantum system from very few measurements
    <https://doi.org/10.1038/s41567-020-0932-7>`_

    The numpy.array shape is `(2, 2)`.
    """
    classical_registers_actually: list[int]
    """The list of the selected_classical_registers."""
    taking_time: float
    """The time taken for the calculation."""
    snapshots: int
    """The number of random basis for classical shadow."""
    shots: int
    """The number of shots."""


class ClassicalShadowMeanRho(ClassicalShadowBasic):
    """The esitimations of the classical shadow from classical snapshots.

    Here, we use the notations that use in the supplementary material of
    `Predicting many properties of a quantum system from very few measurements
    <https://doi.org/10.1038/s41567-020-0932-7>`_

    """

    mean_of_rho: np.ndarray[tuple[int, ...], np.dtype[np.complex128]]
    """The mean of single classical snapshots."""


class EstimationOfObservable(TypedDict):
    """The esitimations of the classical shadow from classical snapshots.

    Here, we use the notations that use in the supplementary material of
    `Predicting many properties of a quantum system from very few measurements
    <https://doi.org/10.1038/s41567-020-0932-7>`_

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

    where :math:`O` is the any operator, and :math:`n` is the number of qubits.
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


class ClassicalShadowEstimation(ClassicalShadowBasic, EstimationOfObservable):
    """The esitimations of the classical shadow from classical snapshots.

    Here, we use the notations that use in the supplementary material of
    `Predicting many properties of a quantum system from very few measurements
    <https://doi.org/10.1038/s41567-020-0932-7>`_

    """


class ClassicalShadowPurity(ClassicalShadowBasic):
    """The expectation value of Rho."""

    purity: Union[float, np.float64]
    """The purity calculated by classical shadow."""
    entropy: Union[float, np.float64]
    """The entropy calculated by classical shadow."""
    purity_value_kind: Union[PurityValueKind, str]
    """The kind of purity value calculation.
    This will depend on the rho_method and trace_method.
    
    If it is not one of the defined kinds, it will be "unknown".
    """


class ClassicalShadowComplex(
    ClassicalShadowEstimation, ClassicalShadowMeanRho, ClassicalShadowPurity
):
    """The expectation value of Rho and the purity calculated by classical shadow."""
