"""ShadowUnveil - Experiment (:mod:`qurry.qurrent.classical_shadow.experiment`)"""

from typing import Union, Optional, Type, Any, Literal, TypedDict
from collections.abc import Iterable, Hashable
import tqdm
import numpy as np
from numpy.random import default_rng

from qiskit import QuantumCircuit

from .analysis import ShadowUnveilAnalysis
from .arguments import ShadowUnveilArguments, SHORT_NAME
from .utils import circuit_method_core
from ..randomized_measure.utils import bitstring_mapping_getter
from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...qurrium.utils.random_unitary import check_input_for_experiment
from ...process.utils import qubit_mapper
from ...process.classical_shadow import (
    classical_shadow_complex,
    ClassicalShadowComplex,
    RhoMCoreMethod,
    TraceRhoMethod,
    AllTraceRhoMethod,
    DEFAULT_ALL_TRACE_RHO_METHOD,
    set_cpu_only,
)
from ...process.classical_shadow.rho_m_core import JAX_AVAILABLE
from ...tools import ParallelManager, set_pbar_description
from ...exceptions import RandomizedMeasureUnitaryOperatorNotFullCovering


class ShadowUnveilExperiment(ExperimentPrototype[ShadowUnveilArguments, ShadowUnveilAnalysis]):
    """The instance of experiment."""

    __name__ = "ShadowUnveilExperiment"

    @property
    def arguments_instance(self) -> Type[ShadowUnveilArguments]:
        """The arguments instance for this experiment."""
        return ShadowUnveilArguments

    @property
    def analysis_instance(self) -> Type[ShadowUnveilAnalysis]:
        """The analysis instance for this experiment."""
        return ShadowUnveilAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        times: int = 100,
        measure: Optional[Union[list[int], tuple[int, int], int]] = None,
        unitary_loc: Optional[Union[list[int], tuple[int, int], int]] = None,
        unitary_loc_not_cover_measure: bool = False,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        **custom_kwargs: Any,
    ) -> tuple[ShadowUnveilArguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            times (int, optional):
                The number of random unitary operator. Defaults to 100.
                It will denote as `N_U` in the experiment name.
            measure (Optional[Union[list[int], tuple[int, int], int]], optional):
                The measure range. Defaults to None.
            unitary_loc (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator. Defaults to None.
            unitary_loc_not_cover_measure (bool, optional):
                Confirm that not all unitary operator are covered by the measure.
                If True, then close the warning.
                Defaults to False.
            random_unitary_seeds (Optional[dict[int, dict[int, int]]], optional):
                The seeds for all random unitary operator.
                This argument only takes input as type of `dict[int, dict[int, int]]`.
                The first key is the index for the random unitary operator.
                The second key is the index for the qubit.

                .. code-block:: python
                    {
                        0: {0: 1234, 1: 5678},
                        1: {0: 2345, 1: 6789},
                        2: {0: 3456, 1: 7890},
                    }

                If you want to generate the seeds for all random unitary operator,
                you can use the function `generate_random_unitary_seeds`
                in `qurry.qurrium.utils.random_unitary`.

                .. code-block:: python
                    from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds
                    random_unitary_seeds = generate_random_unitary_seeds(100, 2)

            custom_kwargs (Any):
                The custom parameters.

        Raises:
            ValueError: If the number of targets is not one.
            TypeError: If times is not an integer.
            ValueError: If the range of measure is not in the range of unitary_loc.

        Returns:
            tuple[EntropyMeasureRandomizedArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) > 1:
            raise ValueError("The number of target circuits should be only one.")
        if not isinstance(times, int):
            raise TypeError(f"times should be an integer, but got {times} as type {type(times)}.")
        if times < 2:
            raise ValueError(
                "times should be greater than 1 for classical shadow "
                + f"on the calculation of entangled entropy, but got {times}."
            )

        target_key, target_circuit = targets[0]
        actual_qubits = target_circuit.num_qubits

        registers_mapping = qubit_mapper(actual_qubits, measure)
        qubits_measured = list(registers_mapping)

        unitary_located = list(qubit_mapper(actual_qubits, unitary_loc))
        measured_but_not_unitary_located = [
            qi for qi in qubits_measured if qi not in unitary_located
        ]
        if len(measured_but_not_unitary_located) > 0 and not unitary_loc_not_cover_measure:
            raise RandomizedMeasureUnitaryOperatorNotFullCovering(
                f"Some qubits {measured_but_not_unitary_located} are measured "
                + "but not random unitary located. "
                + f"unitary_loc: {unitary_loc}, measure: {measure} "
                + "If you are sure about this, you can set `unitary_loc_not_cover_measure=True` "
                + "to close this warning."
            )

        exp_name = f"{exp_name}.N_U_{times}.{SHORT_NAME}"

        check_input_for_experiment(times, len(unitary_located), random_unitary_seeds)

        # pylint: disable=protected-access
        return ShadowUnveilArguments._filter(
            exp_name=exp_name,
            target_keys=[target_key],
            times=times,
            qubits_measured=qubits_measured,
            registers_mapping=registers_mapping,
            actual_num_qubits=actual_qubits,
            unitary_located=unitary_located,
            random_unitary_seeds=random_unitary_seeds,
            **custom_kwargs,
        )
        # pylint: enable=protected-access

    @classmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: ShadowUnveilArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (EntropyMeasureRandomizedArguments):
                The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the side products.
        """
        side_product = {}

        set_pbar_description(pbar, f"Preparing {arguments.times} random unitary.")

        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)

        assert arguments.unitary_located is not None, "unitary_located should be specified."
        random_unitary_ids_array = np.random.randint(
            0, 3, size=(arguments.times, len(arguments.unitary_located))
        ).tolist()
        random_unitary_ids = {
            n_u_i: {
                n_u_qi: (
                    random_unitary_ids_array[n_u_i][seed_i]
                    if arguments.random_unitary_seeds is None
                    else int(
                        default_rng(arguments.random_unitary_seeds[n_u_i][seed_i]).integers(0, 3)
                    )
                )
                for seed_i, n_u_qi in enumerate(arguments.unitary_located)
            }
            for n_u_i in range(arguments.times)
        }

        set_pbar_description(pbar, f"Building {arguments.times} circuits.")
        assert arguments.registers_mapping is not None, "registers_mapping should be not None."
        if multiprocess:
            pool = ParallelManager()
            circ_list = pool.starmap(
                circuit_method_core,
                [
                    (
                        n_u_i,
                        target_circuit,
                        target_key,
                        arguments.exp_name,
                        arguments.registers_mapping,
                        random_unitary_ids[n_u_i],
                    )
                    for n_u_i in range(arguments.times)
                ],
            )
        else:
            circ_list = [
                circuit_method_core(
                    n_u_i,
                    target_circuit,
                    target_key,
                    arguments.exp_name,
                    arguments.registers_mapping,
                    random_unitary_ids[n_u_i],
                )
                for n_u_i in range(arguments.times)
            ]

        set_pbar_description(pbar, "Writing 'random_unitary_ids'.")
        side_product["random_unitary_ids"] = random_unitary_ids

        return circ_list, side_product

    def analyze(
        self,
        selected_qubits: Optional[Iterable[int]] = None,
        # estimation of given operators
        given_operators: Optional[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
        ] = None,
        accuracy_prob_comp_delta: float = 0.01,
        max_shadow_norm: Optional[float] = None,
        # other config
        rho_method: RhoMCoreMethod = "numpy_precomputed",
        trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
        estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
        counts_used: Optional[Iterable[int]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> ShadowUnveilAnalysis:
        r"""Calculate entangled entropy with more information combined.

        Args:
            selected_qubits (Optional[Iterable[int]], optional):
                The selected qubits. Defaults to None.

            given_operators (Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]):
                The list of the operators to estimate. Defaults to None.
            accuracy_prob_comp_delta (float, optional):
                The accuracy probability component delta. Defaults to 0.01.
            max_shadow_norm (Optional[float], optional):
                The maximum shadow norm. Defaults to None.
                If it is None, it will be calculated by the largest shadow norm upper bound.
                If it is not None, it must be a positive float number.
                It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

            rho_method (RhoMCoreMethod, optional):
                The method to use for the calculation. Defaults to "numpy_precomputed".
                It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
                - "numpy": Use Numpy to calculate the rho_m.
                - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
                - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
                Currently, "numpy_precomputed" is the best option for performance.
            trace_method (Union[SingleTraceRhoMethod, AllTraceRhoMethod], optional):
                The method to calculate the trace of Rho square.
                - "trace_of_matmul":
                    Use np.trace(np.matmul(rho_m1, rho_m2))
                    to calculate the each summation item in `rho_m_list`.
                - "quick_trace_of_matmul" or "einsum_ij_ji":
                    Use np.einsum("ij,ji", rho_m1, rho_m2)
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_aij_bji_to_ab_numpy":
                    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
                - "einsum_aij_bji_to_ab_jax":
                    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            estimate_trace_method (AllTraceRhoMethod, optional):
                The method to calculate the trace for searching esitmator.
                - "einsum_aij_bji_to_ab_numpy":
                    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
                - "einsum_aij_bji_to_ab_jax":
                    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used. Defaults to None.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            ShadowUnveilAnalysis: The result of the analysis.
        """

        if selected_qubits is None:
            raise ValueError("selected_qubits should be specified.")
        assert self.args.registers_mapping is not None, "registers_mapping should be not None."

        assert (
            "random_unitary_ids" in self.beforewards.side_product
        ), "The side product 'random_unitary_ids' should be in the side product of the beforewards."
        if len(self.beforewards.side_product["random_unitary_ids"]) != self.args.times:
            raise ValueError(
                f"The number of random unitary ids should be {self.args.times}, "
                + f"but got {len(self.beforewards.side_product['random_unitary_ids'])}."
            )
        random_unitary_ids = {
            int(k): {int(k2): int(v2) for k2, v2 in v.items()}
            for k, v in self.beforewards.side_product["random_unitary_ids"].items()
        }
        assert isinstance(
            self.args.registers_mapping, dict
        ), f"registers_mapping {self.args.registers_mapping} is not dict."

        if isinstance(counts_used, Iterable):
            if max(counts_used) >= len(self.afterwards.counts):
                raise ValueError(
                    "counts_used should be less than "
                    f"{len(self.afterwards.counts)}, but get {max(counts_used)}."
                )
            counts = [self.afterwards.counts[i] for i in counts_used]
        elif counts_used is not None:
            raise ValueError(f"counts_used should be Iterable, but get {type(counts_used)}.")
        else:
            counts = self.afterwards.counts

        bitstring_mapping, final_mapping = bitstring_mapping_getter(
            counts, self.args.registers_mapping
        )

        selected_qubits = [qi % self.args.actual_num_qubits for qi in selected_qubits]
        if len(set(selected_qubits)) != len(selected_qubits):
            raise ValueError(
                f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
            )

        random_unitary_ids_classical_registers = {
            n_u_i: {ci: random_unitary_id[n_u_qi] for n_u_qi, ci in final_mapping.items()}
            for n_u_i, random_unitary_id in random_unitary_ids.items()
        }

        qs = self.quantities(
            shots=self.commons.shots,
            counts=counts,
            random_unitary_ids=random_unitary_ids_classical_registers,
            selected_classical_registers=[final_mapping[qi] for qi in selected_qubits],
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
            pbar=pbar,
        )

        serial = len(self.reports)
        analysis = self.analysis_instance(
            serial=serial,
            num_qubits=self.args.actual_num_qubits,
            selected_qubits=selected_qubits,
            registers_mapping=self.args.registers_mapping,
            bitstring_mapping=bitstring_mapping,
            shots=self.commons.shots,
            unitary_located=self.args.unitary_located,
            counts_used=counts_used,
            **qs,
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        counts: Optional[list[dict[str, int]]] = None,
        random_unitary_ids: Optional[dict[int, dict[int, Union[Literal[0, 1, 2], int]]]] = None,
        selected_classical_registers: Optional[Iterable[int]] = None,
        # estimation of given operators
        given_operators: Optional[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]
        ] = None,
        accuracy_prob_comp_delta: float = 0.01,
        max_shadow_norm: Optional[float] = None,
        # other config
        rho_method: RhoMCoreMethod = "numpy_precomputed",
        trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
        estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> ClassicalShadowComplex:
        r"""Randomized entangled entropy with complex.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The list of the counts.
            random_unitary_ids (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
                The shadow direction of the unitary operators.
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

            rho_method (RhoMCoreMethod, optional):
                The method to use for the calculation. Defaults to "numpy_precomputed".
                It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
                - "numpy": Use Numpy to calculate the rho_m.
                - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
                - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
                Currently, "numpy_precomputed" is the best option for performance.
            trace_method (TraceRhoMethod, optional):
                The method to calculate the trace of Rho square.
                - "trace_of_matmul":
                    Use np.trace(np.matmul(rho_m1, rho_m2))
                    to calculate the each summation item in `rho_m_list`.
                - "quick_trace_of_matmul" or "einsum_ij_ji":
                    Use np.einsum("ij,ji", rho_m1, rho_m2)
                    to calculate the each summation item in `rho_m_list`.
                - "einsum_aij_bji_to_ab_numpy":
                    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
                - "einsum_aij_bji_to_ab_jax":
                    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            estimate_trace_method (AllTraceRhoMethod, optional):
                The method to calculate the trace for searching esitmator.
                - "einsum_aij_bji_to_ab_numpy":
                    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
                - "einsum_aij_bji_to_ab_jax":
                    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            ClassicalShadowComplex: The result of the classical shadow.
        """

        if shots is None or counts is None:
            raise ValueError("shots and counts should be specified.")
        if random_unitary_ids is None:
            raise ValueError("random_unitary_ids should be specified.")
        if selected_classical_registers is None:
            raise ValueError("selected_classical_registers should be specified.")

        return classical_shadow_complex(
            shots=shots,
            counts=counts,
            random_unitary_um=random_unitary_ids,
            selected_classical_registers=selected_classical_registers,
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
            pbar=pbar,
        )

    def outside_analysis_recover(
        self,
        analysis: ShadowUnveilAnalysis,
    ) -> ShadowUnveilAnalysis:
        """Recover the analysis from the outside.

        Args:
            analysis (ShadowUnveilAnalysis):
                The analysis to recover.

        Returns:
            ShadowUnveilAnalysis: The recovered analysis.
        """

        if analysis.serial in self.reports:
            analysis.serial = len(self.reports)

        self.reports[analysis.serial] = analysis
        return analysis


class OutsideAnalyzeInput(TypedDict):
    """The input for the outside analyze."""

    exp_id: str
    # for analze
    shots: int
    counts: list[dict[str, int]]
    random_unitary_ids: dict[int, dict[int, Union[Literal[0, 1, 2], int]]]
    selected_classical_registers: Iterable[int]
    # for analysis input
    num_qubits: int
    selected_qubits: list[int]
    registers_mapping: dict[int, int]
    bitstring_mapping: dict[int, int]
    unitary_located: list[int]
    # estimation of given operators
    given_operators: Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]
    accuracy_prob_comp_delta: float
    max_shadow_norm: Optional[float]
    # setup for running
    serial: int
    rho_method: RhoMCoreMethod
    trace_method: TraceRhoMethod
    estimate_trace_method: AllTraceRhoMethod
    counts_used: Optional[Iterable[int]]


def quantities_input_collecter(
    current_exps: ShadowUnveilExperiment,
    # analysis inputs
    selected_qubits: Optional[Iterable[int]] = None,
    # estimation of given operators
    given_operators: Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]] = None,
    accuracy_prob_comp_delta: float = 0.01,
    max_shadow_norm: Optional[float] = None,
    # other config
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    counts_used: Optional[Iterable[int]] = None,
) -> OutsideAnalyzeInput:
    r"""Collect the inputs for the quantities.

    Args:
        current_exps (ShadowUnveilExperiment):
            The current experiment instance.
        selected_qubits (Optional[Iterable[int]], optional):
            The selected qubits. Defaults to None.

        given_operators (Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]):
            The list of the operators to estimate. Defaults to None.
        accuracy_prob_comp_delta (float, optional):
            The accuracy probability component delta. Defaults to 0.01.
        max_shadow_norm (Optional[float], optional):
            The maximum shadow norm. Defaults to None.
            If it is None, it will be calculated by the largest shadow norm upper bound.
            If it is not None, it must be a positive float number.
            It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.

        backend (PostProcessingBackendLabel, optional):
            The backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        trace_method (Union[SingleTraceRhoMethod, AllTraceRhoMethod], optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul":
                Use np.trace(np.matmul(rho_m1, rho_m2))
                to calculate the each summation item in `rho_m_list`.
            - "quick_trace_of_matmul" or "einsum_ij_ji":
                Use np.einsum("ij,ji", rho_m1, rho_m2)
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        estimate_trace_method (AllTraceRhoMethod, optional):
            The method to calculate the trace for searching esitmator.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        counts_used (Optional[Iterable[int]], optional):
            The index of the counts used. Defaults to None.

    Returns:
        OutsideAnalyzeInput: The inputs for the quantities.
    """

    if selected_qubits is None:
        raise ValueError("selected_qubits should be specified.")
    assert current_exps.args.registers_mapping is not None, "registers_mapping should be not None."

    assert (
        "random_unitary_ids" in current_exps.beforewards.side_product
    ), "The side product 'random_unitary_ids' should be in the side product of the beforewards."
    random_unitary_ids = current_exps.beforewards.side_product["random_unitary_ids"]
    assert isinstance(
        current_exps.args.registers_mapping, dict
    ), f"registers_mapping {current_exps.args.registers_mapping} is not dict."

    if isinstance(counts_used, Iterable):
        if max(counts_used) >= len(current_exps.afterwards.counts):
            raise ValueError(
                "counts_used should be less than "
                f"{len(current_exps.afterwards.counts)}, but get {max(counts_used)}."
            )
        counts = [current_exps.afterwards.counts[i] for i in counts_used]
    elif counts_used is not None:
        raise ValueError(f"counts_used should be Iterable, but get {type(counts_used)}.")
    else:
        counts = current_exps.afterwards.counts

    bitstring_mapping, final_mapping = bitstring_mapping_getter(
        counts, current_exps.args.registers_mapping
    )

    selected_qubits = [qi % current_exps.args.actual_num_qubits for qi in selected_qubits]
    if len(set(selected_qubits)) != len(selected_qubits):
        raise ValueError(
            f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
        )

    random_unitary_ids_classical_registers = {
        n_u_i: {ci: random_unitary_id[n_u_qi] for n_u_qi, ci in final_mapping.items()}
        for n_u_i, random_unitary_id in random_unitary_ids.items()
    }

    serial = len(current_exps.reports)
    assert current_exps.args.unitary_located is not None, "unitary_located should be specified."

    return {
        "exp_id": current_exps.exp_id,
        # for analyze
        "shots": current_exps.commons.shots,
        "counts": counts,
        "random_unitary_ids": random_unitary_ids_classical_registers,
        "selected_classical_registers": [final_mapping[qi] for qi in selected_qubits],
        # for analysis instance
        "num_qubits": current_exps.args.actual_num_qubits,
        "selected_qubits": selected_qubits,
        "registers_mapping": current_exps.args.registers_mapping,
        "bitstring_mapping": bitstring_mapping,
        "unitary_located": current_exps.args.unitary_located,
        # estimation of given operators
        "given_operators": given_operators,
        "accuracy_prob_comp_delta": accuracy_prob_comp_delta,
        "max_shadow_norm": max_shadow_norm,
        # setup for running
        "serial": serial,
        "rho_method": rho_method,
        "trace_method": trace_method,
        "estimate_trace_method": estimate_trace_method,
        "counts_used": counts_used,
    }


def outside_analyze(
    exp_id: str,
    # for analyze
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_ids: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
    # for analysis instance
    num_qubits: int,
    selected_qubits: list[int],
    registers_mapping: dict[int, int],
    bitstring_mapping: dict[int, int],
    unitary_located: list[int],
    # estimation of given operators
    given_operators: Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]],
    accuracy_prob_comp_delta: float,
    max_shadow_norm: Optional[float],
    # setup for running
    serial: int,
    rho_method: RhoMCoreMethod = "numpy_precomputed",
    trace_method: TraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    estimate_trace_method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
    counts_used: Optional[Iterable[int]] = None,
) -> tuple[str, ShadowUnveilAnalysis]:
    r"""Randomized entangled entropy with complex.

    Args:
        exp_id (str):
            The ID of the experiment.

        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_ids (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.

        num_qubits (int):
            The number of qubits.
        selected_qubits (list[int]):
            The selected qubits.
        registers_mapping (dict[int, int]):
            The mapping of the index of selected qubits to the index of the classical register.
        bitstring_mapping (dict[str, int]):
            The mapping of the bitstring to the index of the classical register.
        unitary_located (list[int]):
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

        serial (int):
            The serial number of the experiment.
        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy_precomputed".
            It can be either "numpy", "numpy_precomputed", "jax_flatten", or "numpy_flatten".
            - "numpy": Use Numpy to calculate the rho_m.
            - "numpy_precomputed": Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_flatten": Use Numpy to calculate the rho_m with a flattening workflow.
            Currently, "numpy_precomputed" is the best option for performance.
        trace_method (TraceRhoMethod, optional):
            The method to calculate the trace of Rho square.
            - "trace_of_matmul":
                Use np.trace(np.matmul(rho_m1, rho_m2))
                to calculate the each summation item in `rho_m_list`.
            - "quick_trace_of_matmul" or "einsum_ij_ji":
                Use np.einsum("ij,ji", rho_m1, rho_m2)
                to calculate the each summation item in `rho_m_list`.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        estimate_trace_method (AllTraceRhoMethod, optional):
            The method to calculate the trace for searching esitmator.
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
        backend (PostProcessingBackend, optional):
            Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
        counts_used (Optional[Iterable[int]], optional):
            The index of the counts used. Defaults to None.

    Returns:
        tuple[str, ShadowUnveilAnalysis]:
            The ID of the experiment and the result of the classical shadow.
    """

    if JAX_AVAILABLE:
        set_cpu_only()

    qs = classical_shadow_complex(
        shots=shots,
        counts=counts,
        random_unitary_um=random_unitary_ids,
        selected_classical_registers=selected_classical_registers,
        # estimation of given operators
        given_operators=given_operators,
        accuracy_prob_comp_delta=accuracy_prob_comp_delta,
        max_shadow_norm=max_shadow_norm,
        # other config
        rho_method=rho_method,
        trace_method=trace_method,
        estimate_trace_method=estimate_trace_method,
        pbar=None,
    )

    analysis = ShadowUnveilAnalysis(
        shots=shots,
        # for analysis input
        num_qubits=num_qubits,
        selected_qubits=selected_qubits,
        registers_mapping=registers_mapping,
        bitstring_mapping=bitstring_mapping,
        unitary_located=unitary_located,
        # setup for running
        serial=serial,
        counts_used=counts_used,
        **qs,
    )

    return exp_id, analysis


def outside_analyze_wrapper(
    all_arguments: OutsideAnalyzeInput,
) -> tuple[str, ShadowUnveilAnalysis]:
    """Wrapper for the outside analyze.

    Args:
        all_arguments (OutsideAnalyzeInput):
            The arguments for the outside analyze.

    Returns:
        tuple[str, ShadowUnveilAnalysis]:
            The ID of the experiment and the result of the classical shadow.
    """
    return outside_analyze(**all_arguments)
