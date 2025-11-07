"""ShadowUnveil - Analysis (:mod:`qurry.qurrent.classical_shadow.analysis`)"""

from typing import Optional, NamedTuple, Iterable, Any, Type, Union, Literal
import numpy as np

from ...qurrium.analysis import AnalysisPrototype
from ...qurrium.utils import bitstring_mapping_getter
from ...process.utils import counts_list_recount_pyrust
from ...process.classical_shadow import (
    PurityValueKind,
    classical_shadow_complex,
    ClassicalShadowComplex,
    RhoMethodType,
    DEFAULT_RHO_METHOD,
    TraceMethodType,
    DEFAULT_TRACE_METHOD,
    ListTraceMethodType,
    DEFAULT_LIST_TRACE_METHOD,
)


class SUAnalysisInput(NamedTuple):
    """To set the analysis."""

    shots: int
    """The number of shots."""
    snapshots: int
    """The number of random basis for classical shadow."""
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

    counts_used: Optional[Iterable[int]] = None
    """The index of the counts used."""


class SUAnalysisContent(NamedTuple):
    """The content of the analysis."""

    average_classical_snapshots_rho: dict[int, np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
    """The dictionary of Rho M."""
    classical_registers_actually: list[int]
    """The list of the selected_classical_registers."""
    taking_time: float
    """The time taken for the calculation."""
    rho_method: RhoMethodType
    """The method to calculate the rho."""
    # The mean of Rho
    mean_of_rho: np.ndarray[tuple[int, int], np.dtype[np.complex128]]
    """The expectation value of Rho."""

    # esitimation of given operators
    given_operators: list[np.ndarray[tuple[int, ...], np.dtype[np.complex128]]]
    """The list of the operators to estimate."""
    estimate_of_given_operators: list[np.complex128]
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
    estimate_trace_method: ListTraceMethodType
    """The method to calculate the trace for searching estimators."""

    # The trace of Rho square
    purity: float
    """The purity calculated by classical shadow."""
    entropy: float
    """The entropy calculated by classical shadow."""
    purity_value_kind: Union[PurityValueKind, str]
    """The kind of purity value calculation.
    This will depend on the rho_method and trace_method.

    If it is not one of the defined kinds, it will be "unknown".
    """
    trace_method: TraceMethodType
    """The method to calculate the trace of Rho."""

    def __repr__(self):
        return f"SUAnalysisContent(purity={self.purity}, entropy={self.entropy}, and others)"


FIELDS_REMAPPING = {
    "rho_m_dict": "average_classical_snapshots_rho",
    "expect_rho": "mean_of_rho",
}
"""Remapping of fields from old in 0.12 to new names since 0.13.
The keys are the old field names and the values are the new field names.
"""

NEW_FIELDS_DEFAULTS = {
    "average_classical_snapshots_rho": {},
    "mean_of_rho": np.zeros((1, 1), dtype=np.complex128),
    "classical_registers_actually": [],
    "rho_method": "unknown",
    "taking_time": 0.0,
    "purity": np.nan,
    "entropy": np.nan,
    "purity_value_kind": "unknown",
    "trace_method": "unknown",
    "estimate_of_given_operators": [],
    "corresponding_rhos": [],
    "accuracy_prob_comp_delta": np.nan,
    "num_of_estimators_k": 0,
    "accuracy_predict_epsilon": np.nan,
    "maximum_shadow_norm": np.nan,
    "estimate_trace_method": "unknown",
}
"""Default values for new fields introduced in 0.13."""


class ShadowUnveilAnalysis(AnalysisPrototype[SUAnalysisInput, SUAnalysisContent]):
    """The container for the analysis of
    :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`."""

    __name__ = "SUAnalysis"

    @classmethod
    def input_type(cls) -> Type[SUAnalysisInput]:
        """The type of the input for the analysis."""
        return SUAnalysisInput

    @classmethod
    def quantities(
        cls,
        shots: int,
        counts: list[dict[str, int]],
        random_basis_array: Optional[list[list[Union[Literal[0, 1, 2], int]]]] = None,
        selected_classical_registers: Optional[Iterable[int]] = None,
        # estimation of given operators
        given_operators: Optional[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
        ] = None,
        accuracy_prob_comp_delta: float = 0.01,
        max_shadow_norm: Optional[float] = None,
        # other config
        rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
        trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
        estimate_trace_method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
    ) -> ClassicalShadowComplex:
        r"""Calculate the classical shadow quantities.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The list of the counts.
            random_basis_array (list[list[Union[Literal[0, 1, 2], int]]]):
                The random basis for classical shadow.
            selected_classical_registers (Iterable[int]):
                The list of **the index of the selected_classical_registers**.

            given_operators (Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]):
                The list of the operators to estimate. Defaults to None.
            accuracy_prob_comp_delta (float, optional):
                The accuracy probability component delta. Defaults to 0.01.
            max_shadow_norm (Optional[float], optional):
                The maximum shadow norm. Defaults to None.
                If it is None, it will be calculated by the largest shadow norm upper bound.
                If it is not None, it must be a positive float number.
                It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

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

                - Skip calculation of trace:
                    - "skip_trace": Skip the trace calculation and return NaN.

                The default method is "bitwise_py", which is the fastest option.
            estimate_trace_method (ListTraceMethodType, optional):
                The method to use for the calculation.

                - "einsum_aij_bji_to_ab_numpy":
                    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is not available.
                - "einsum_aij_bji_to_ab_jax":
                    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho.

                Defaults to DEFAULT_LIST_TRACE_METHOD.

            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            ClassicalShadowComplex: The result of the classical shadow.
        """

        if random_basis_array is None:
            raise ValueError("random_unitary_ids should be specified.")
        if selected_classical_registers is None:
            raise ValueError("selected_classical_registers should be specified.")

        return classical_shadow_complex(
            shots=shots,
            counts=counts,
            random_basis_array=random_basis_array,
            selected_classical_registers=selected_classical_registers,
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
        )

    @classmethod
    def make(
        cls,
        *,
        serial: int,
        shots: int,
        counts: list[dict[str, int]],
        selected_qubits: Optional[Iterable[int]] = None,
        registers_mapping: Optional[dict[int, int]] = None,
        snapshots: Optional[int] = None,
        num_qubits: Optional[int] = None,
        random_basis: Optional[dict[int, dict[int, int]]] = None,
        unitary_located: Optional[list[int]] = None,
        # estimation of given operators
        given_operators: Optional[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
        ] = None,
        accuracy_prob_comp_delta: float = 0.01,
        max_shadow_norm: Optional[float] = None,
        # other config
        rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
        trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
        estimate_trace_method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
        counts_used: Optional[Iterable[int]] = None,
    ) -> "ShadowUnveilAnalysis":
        """Make an analysis instance of ShadowUnveil.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The list of the counts.

            selected_qubits (Optional[Iterable[int]]):
                The selected qubits.
            registers_mapping (Optional[dict[int, int]]):
                The mapping of the classical registers with quantum registers.
            snapshots (Optional[int]):
                The number of random basis for classical shadow.
            num_qubits (Optional[int]):
                The number of qubits.
            random_basis (Optional[dict[int, dict[int, int]]]):
                The random basis for classical shadow.
            unitary_located (Optional[list[int]]):
                The range of the unitary operator.

            given_operators (Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]):
                The list of the operators to estimate. Defaults to None.
            accuracy_prob_comp_delta (float, optional):
                The accuracy probability component delta. Defaults to 0.01.
            max_shadow_norm (Optional[float], optional):
                The maximum shadow norm. Defaults to None.
                If it is None, it will be calculated by the largest shadow norm upper bound.
                If it is not None, it must be a positive float number.
                It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

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

                - Skip calculation of trace:
                    - "skip_trace": Skip the trace calculation and return NaN.

                The default method is "bitwise_py", which is the fastest option.
            estimate_trace_method (ListTraceMethodType, optional):
                The method to use for the calculation.

                - "einsum_aij_bji_to_ab_numpy":
                    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho
                    if JAX is not available.
                - "einsum_aij_bji_to_ab_jax":
                    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                    This is the fastest implementation to calculate the trace of Rho.

                Defaults to DEFAULT_LIST_TRACE_METHOD.

            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used. Defaults to None.
        Returns:
            The analysis instance.

        """

        if selected_qubits is None:
            raise ValueError("selected_qubits should be specified.")
        if registers_mapping is None:
            raise ValueError("registers_mapping should be specified.")
        if snapshots is None:
            raise ValueError("snapshots should be specified.")
        if num_qubits is None:
            raise ValueError("num_qubits should be specified.")
        if random_basis is None:
            raise ValueError("random_basis should be specified.")
        if unitary_located is None:
            raise ValueError("unitary_located should be specified.")

        if len(random_basis) != snapshots:
            raise ValueError(
                f"The number of random basis should be {snapshots}, "
                + f"but got {len(random_basis)}."
            )
        if not isinstance(registers_mapping, dict):
            raise ValueError(
                f"The registers_mapping should be dict, but got {type(registers_mapping)}."
            )
        if isinstance(counts_used, Iterable):
            if max(counts_used) >= len(counts):
                raise ValueError(
                    f"counts_used should be less than {len(counts)}, but get {max(counts_used)}."
                )
            counts = [counts[i] for i in counts_used]

        bitstring_mapping, final_mapping = bitstring_mapping_getter(counts, registers_mapping)

        counts = counts_list_recount_pyrust(
            counts, len(next(iter(counts[0].keys()))), list(final_mapping.values())
        )

        selected_qubits = [qi % num_qubits for qi in selected_qubits]
        if len(set(selected_qubits)) != len(selected_qubits):
            raise ValueError(
                f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
            )
        selected_clregs_sorted = sorted([registers_mapping[qi] for qi in selected_qubits])
        all_clregs = sorted(registers_mapping.values())

        random_basis_with_clreg_index = []
        for i in range(len(random_basis)):
            tmp = {ci: random_basis[i][n_u_qi] for n_u_qi, ci in registers_mapping.items()}
            random_basis_with_clreg_index.append([tmp[j] for j in all_clregs])

        qs = cls.quantities(
            shots=shots,
            counts=counts,
            random_basis_array=random_basis_with_clreg_index,
            selected_classical_registers=selected_clregs_sorted,
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
        )

        return ShadowUnveilAnalysis(
            serial=serial,
            # input
            num_qubits=num_qubits,
            selected_qubits=selected_qubits,
            registers_mapping=registers_mapping,
            bitstring_mapping=bitstring_mapping,
            unitary_located=unitary_located,
            counts_used=counts_used,
            # content
            **qs,
        )

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
        if "expect_rho" in main or "rho_m_dict" in side:
            main["mean_of_rho"] = main.pop("expect_rho")
            side["average_classical_snapshots_rho"] = side.pop("rho_m_dict")
            for k, v in NEW_FIELDS_DEFAULTS.items():
                if k not in main:
                    main[k] = v

        if "snapshots" not in main["input"]:
            main["input"]["snapshots"] = len(side["average_classical_snapshots_rho"])

        if "methods_used" in main:
            rho_method, trace_method = main.pop("methods_used")
            main["rho_method"] = rho_method
            main["trace_method"] = trace_method
            main["estimate_trace_method"] = "unknown"

        return main, side

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return ["average_classical_snapshots_rho", "corresponding_rhos"]
