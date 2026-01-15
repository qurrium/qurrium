"""ShadowUnveil - Qurrium (:mod:`qurry.qurries.classical_shadow.qurry`)"""

from typing import Union, Optional, Literal
from collections.abc import Iterable
import warnings
from pathlib import Path
import tqdm
import numpy as np
import numpy.typing as npt

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .arguments import SHORT_NAME, ACRONYM, SUMeasureArgs, SUOutputArgs
from .analysis import SUAnalyzeArgs
from .experiment import SUExperiment
from ...qurrium import (
    QurriumPrototype,
    RunArgsType,
    TranspileArgs,
    PassManagerType,
    SpecificAnalyzeArgs,
    WCKeyable,
)
from ...process.utils import QubitSelectionType
from ...process.classical_shadow import (
    JAX_AVAILABLE,
    ShadowBasisType,
    RhoMethodType,
    DEFAULT_RHO_METHOD,
    TraceMethodType,
    DEFAULT_TRACE_METHOD,
    ListTraceMethodType,
    DEFAULT_LIST_TRACE_METHOD,
)


class ShadowUnveil(QurriumPrototype[SUExperiment, SUMeasureArgs, SUOutputArgs, SUAnalyzeArgs]):
    r"""Classical Shadow with The Results of Second Order Renyi Entropy.

    Reference:
        -   Predicting many properties of a quantum system from very few measurements -
            Huang, Hsin-Yuan and Kueng, Richard and Preskill, John
            `doi:10.1038/s41567-020-0932-7 <https://doi.org/10.1038/s41567-020-0932-7>`_

        -   The randomized measurement toolbox -
            Elben, Andreas and Flammia, Steven T. and Huang, Hsin-Yuan and Kueng,
            Richard and Preskill, John and Vermersch, Benoît and Zoller, Peter
            `doi:10.1038/s42254-022-00535-2 <https://doi.org/10.1038/s42254-022-00535-2>`_

        .. code-block:: bibtex

            @article{cite-key,
                abstract = {
                    Predicting the properties of complex,
                    large-scale quantum systems is essential for developing quantum technologies.
                    We present an efficient method for constructing an approximate classical
                    description of a quantum state using very few measurements of the state.
                    different properties; order
                    {\$}{\$}{\{}{$\backslash$}mathrm{\{}log{\}}{\}}{$\backslash$},(M){\$}{\$}
                    measurements suffice to accurately predict M different functions of the state
                    with high success probability. The number of measurements is independent of
                    the system size and saturates information-theoretic lower bounds. Moreover,
                    target properties to predict can be
                    selected after the measurements are completed.
                    We support our theoretical findings with extensive numerical experiments.
                    We apply classical shadows to predict quantum fidelities,
                    entanglement entropies, two-point correlation functions,
                    expectation values of local observables and the energy variance of
                    many-body local Hamiltonians.
                    The numerical results highlight the advantages of classical shadows relative to
                    previously known methods.},
                author = {Huang, Hsin-Yuan and Kueng, Richard and Preskill, John},
                date = {2020/10/01},
                date-added = {2024-12-03 15:00:55 +0800},
                date-modified = {2024-12-03 15:00:55 +0800},
                doi = {10.1038/s41567-020-0932-7},
                id = {Huang2020},
                isbn = {1745-2481},
                journal = {Nature Physics},
                number = {10},
                pages = {1050--1057},
                title = {Predicting many properties of a quantum system from very few measurements},
                url = {https://doi.org/10.1038/s41567-020-0932-7},
                volume = {16},
                year = {2020},
                bdsk-url-1 = {https://doi.org/10.1038/s41567-020-0932-7}
            }

            @article{cite-key,
                abstract = {
                    Programmable quantum simulators and quantum computers are opening unprecedented
                    opportunities for exploring and exploiting the properties of highly entangled
                    complex quantum systems. The complexity of large quantum systems is the source
                    of computational power but also makes them difficult to control precisely or
                    characterize accurately using measured classical data. We review protocols
                    for probing the properties of complex many-qubit systems using measurement
                    schemes that are practical using today's quantum platforms. In these protocols,
                    a quantum state is repeatedly prepared and measured in a randomly chosen basis;
                    then a classical computer processes the measurement outcomes to estimate the
                    desired property. The randomization of the measurement procedure has distinct
                    advantages. For example, a single data set can be used multiple times to pursue
                    a variety of applications, and imperfections in the measurements are mapped to
                    a simplified noise model that can more
                    easily be mitigated. We discuss a range of
                    cases that have already been realized in quantum devices, including Hamiltonian
                    simulation tasks, probes of quantum chaos, measurements of non-local order
                    parameters, and comparison of quantum states produced in distantly separated
                    laboratories. By providing a workable method for translating a complex quantum
                    state into a succinct classical representation that preserves a rich variety of
                    relevant physical properties, the randomized measurement toolbox strengthens our
                    ability to grasp and control the quantum world.},
                author = {
                    Elben, Andreas and Flammia, Steven T. and Huang, Hsin-Yuan and Kueng,
                    Richard and Preskill, John and Vermersch, Beno{\^\i}t and Zoller, Peter},
                date = {2023/01/01},
                date-added = {2024-12-03 15:06:15 +0800},
                date-modified = {2024-12-03 15:06:15 +0800},
                doi = {10.1038/s42254-022-00535-2},
                id = {Elben2023},
                isbn = {2522-5820},
                journal = {Nature Reviews Physics},
                number = {1},
                pages = {9--24},
                title = {The randomized measurement toolbox},
                url = {https://doi.org/10.1038/s42254-022-00535-2},
                volume = {5},
                year = {2023},
                bdsk-url-1 = {https://doi.org/10.1038/s42254-022-00535-2}
            }

    """

    __name__ = "ShadowUnveil"
    short_name = SHORT_NAME
    """The short name of this Qurrium class."""
    acronym = ACRONYM
    """The abbreviation of this Qurrium class."""

    def __post_init__(self):
        """Initialize the class."""
        if JAX_AVAILABLE:
            # pylint: disable=import-outside-toplevel
            import jax

            if not jax.config.values["jax_enable_x64"]:
                jax.config.update("jax_enable_x64", True)
            # pylint: enable=import-outside-toplevel

            if not jax.config.values["jax_enable_x64"]:
                warnings.warn(
                    "JAX is not set to use 64-bit precision, but it should be setup by Qurrium"
                    + "Since we rely on 64-bit precision, please set it to use 64-bit precision. "
                    + "You can set it by `jax.config.update('jax_enable_x64', True)`. "
                    + "Or you can set it in your environment by `export JAX_ENABLE_X64=True`. "
                    + "Otherwise, the results will be inaccurate when use JAX.",
                    RuntimeWarning,
                )

    @property
    def experiment_instance(self) -> type[SUExperiment]:
        """The container class responding to this QurryV5 class."""
        return SUExperiment

    def measure_to_output(
        self,
        wave: Optional[Union[QuantumCircuit, WCKeyable]] = None,
        snapshots: int = 100,
        measure: QubitSelectionType = None,
        unitary_loc: QubitSelectionType = None,
        unitary_loc_not_cover_measure: bool = False,
        shadow_basis_method: Optional[ShadowBasisType] = None,
        random_basis: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: RunArgsType = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager: PassManagerType = None,
        tags: Optional[tuple[str, ...]] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> SUOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave (Union[QuantumCircuit, WCKeyable]):
                The key or the circuit to execute.
            snapshots (int, optional):
                The number of random unitary operator, previously called `times`
                It will denote as :math:`N_U` in the experiment name.
                Defaults to `100`.
            measure (QubitSelectionType, optional):
                The measure range. Defaults to None.
            unitary_loc (QubitSelectionType, optional):
                The range of the unitary operator. Defaults to None.
            unitary_loc_not_cover_measure (bool, optional):
                Whether the range of the unitary operator is not cover the measure range.
                Defaults to `False`.
            shadow_basis_method (Optional[ShadowBasisType], optional):
                The classical shadow basis for sampling. It can be set to
                :class:`~qurry.process.classical_shadow.rho_process.unitary_set.ShadowRandomBasis`
                or
                :class:`~qurry.process.classical_shadow.rho_process.unitary_set.ShadowBasisMethod`.
                Defaults to None, which use the default Pauli basis from
                :meth:`ShadowBasisMethod.`.
            random_basis (Optional[dict[int, dict[int, int]]], optional):
                The random basis for classical shadow.

                This argument only takes input as type of `dict[int, dict[int, int]]`.
                The first key is the index if snapshots.
                The second key is the index for the qubit.

                .. code-block:: python

                    {
                        0: {0: 1, 1: 0},
                        1: {0: 2, 1: 1},
                        2: {0: 0, 1: 2},
                    }

                If you want to generate the seeds for all random unitary operator,
                you can use the function :func:`generate_random_basis`
                in :mod:`qurry.process.classical_shadow.utils`.

                .. code-block:: python

                    from qurry import generate_random_basis

                    random_basis = generate_random_basis(100, [0, 1])


            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`~qiskit.compiler.transpile`.
                Defaults to None.
            passmanager (PassManagerType, optional):
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to None.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
            export (bool, optional):
                Whether to export the experiment. Defaults to False.
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to None.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.

        Returns:
            ShadowUnveilOutputArgs: The output arguments.
        """
        if wave is None:
            raise ValueError("The `wave` must be provided.")

        return {
            "circuits": [wave],
            "snapshots": snapshots,
            "measure": measure,
            "unitary_loc": unitary_loc,
            "unitary_loc_not_cover_measure": unitary_loc_not_cover_measure,
            "random_basis": random_basis,
            "shadow_basis_method": shadow_basis_method,
            "shots": shots,
            "backend": backend,
            "exp_name": exp_name,
            "run_args": run_args,
            "transpile_args": transpile_args,
            "passmanager": passmanager,
            "tags": tags,
            # process tool
            "qasm_version": qasm_version,
            "export": export,
            "save_location": save_location,
            "pbar": pbar,
        }

    def measure(
        self,
        wave: Optional[Union[QuantumCircuit, WCKeyable]] = None,
        snapshots: int = 100,
        measure: QubitSelectionType = None,
        unitary_loc: QubitSelectionType = None,
        unitary_loc_not_cover_measure: bool = False,
        shadow_basis_method: Optional[ShadowBasisType] = None,
        random_basis: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: RunArgsType = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager: PassManagerType = None,
        tags: Optional[tuple[str, ...]] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> str:
        """Execute the experiment.

        Args:
            wave (Union[QuantumCircuit, WCKeyable]):
                The key or the circuit to execute.
            snapshots (int, optional):
                The number of random unitary operator, previously called `times`
                It will denote as :math:`N_U` in the experiment name.
                Defaults to `100`.
            measure (QubitSelectionType, optional):
                The measure range. Defaults to None.
            unitary_loc (QubitSelectionType, optional):
                The range of the unitary operator. Defaults to None.
            unitary_loc_not_cover_measure (bool, optional):
                Whether the range of the unitary operator is not cover the measure range.
                Defaults to `False`.
            shadow_basis_method (Optional[ShadowBasisType], optional):
                The classical shadow basis for sampling. It can be set to
                :class:`~qurry.process.classical_shadow.rho_process.unitary_set.ShadowRandomBasis`
                or
                :class:`~qurry.process.classical_shadow.rho_process.unitary_set.ShadowBasisMethod`.
                Defaults to None, which use the default Pauli basis from
                :meth:`ShadowBasisMethod.`.
            random_basis (Optional[dict[int, dict[int, int]]], optional):
                The random basis for classical shadow.

                This argument only takes input as type of `dict[int, dict[int, int]]`.
                The first key is the index if snapshots.
                The second key is the index for the qubit.

                .. code-block:: python

                    {
                        0: {0: 1, 1: 0},
                        1: {0: 2, 1: 1},
                        2: {0: 0, 1: 2},
                    }

                If you want to generate the seeds for all random unitary operator,
                you can use the function :func:`generate_random_basis`
                in :mod:`qurry.process.classical_shadow.utils`.

                .. code-block:: python

                    from qurry import generate_random_basis

                    random_basis = generate_random_basis(100, [0, 1])


            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`~qiskit.compiler.transpile`.
                Defaults to None.
            passmanager (PassManagerType, optional):
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to None.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
            export (bool, optional):
                Whether to export the experiment. Defaults to False.
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to None.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.

        Returns:
            str: The experiment ID.
        """

        output_args = self.measure_to_output(
            wave=wave,
            snapshots=snapshots,
            measure=measure,
            unitary_loc=unitary_loc,
            unitary_loc_not_cover_measure=unitary_loc_not_cover_measure,
            shadow_basis_method=shadow_basis_method,
            random_basis=random_basis,
            shots=shots,
            backend=backend,
            exp_name=exp_name,
            run_args=run_args,
            transpile_args=transpile_args,
            passmanager=passmanager,
            tags=tags,
            # process tool
            qasm_version=qasm_version,
            export=export,
            save_location=save_location,
            pbar=pbar,
        )

        return self.output(**output_args)

    def multiAnalysis(
        self,
        summoner_id: str,
        *,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: SpecificAnalyzeArgs[SUAnalyzeArgs] = None,
        skip_write: bool = False,
        multiprocess_write: bool = False,
        # analysis arguments
        selected_qubits: Optional[Iterable[int]] = None,
        # estimation of given operators
        given_operators: Optional[list[npt.NDArray[np.complex128]]] = None,
        accuracy_prob_comp_delta: float = 0.01,
        max_shadow_norm: Optional[float] = None,
        # other config
        rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
        trace_method: TraceMethodType = DEFAULT_TRACE_METHOD,
        estimate_trace_method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
        counts_used: Optional[Iterable[int]] = None,
        **analysis_args,
    ) -> tuple[str, str]:
        r"""Run the analysis for multiple experiments.

        Args:
            summoner_id (str): The summoner_id of multimanager.
            analysis_name (str, optional):
                The name of analysis. Defaults to 'report'.
            no_serialize (bool, optional):
                Whether to serialize the analysis. Defaults to False.
            specific_analysis_args(SpecificAnalsisArgs[_RA, optional):
                The specific arguments for analysis. Defaults to None.
            skip_write (bool, optional):
                Whether to skip the file writing during the analysis. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

            selected_qubits (Optional[Iterable[int]], optional):
                The selected qubits. Defaults to None.

            given_operators (Optional[list[npt.NDArray[np.complex128]]]):
                The list of the operators to estimate. Defaults to None.
            accuracy_prob_comp_delta (float, optional):
                The accuracy probability component delta. Defaults to 0.01.
            max_shadow_norm (Optional[float], optional):
                The maximum shadow norm. Defaults to None.
                If it is None, it will be calculated by the largest shadow norm upper bound.
                If it is not None, it must be a positive float number.
                It is :math:`|| O_i - \frac{\text{tr}(O_i)}{2^n} ||_{\text{shadow}}^2` in equation.


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

            analysis_args (Any, optional):
                Other arguments for analysis.

        Returns:
            str: The summoner_id of multimanager.
        """

        return super().multiAnalysis(
            summoner_id=summoner_id,
            analysis_name=analysis_name,
            no_serialize=no_serialize,
            specific_analysis_args=specific_analysis_args,
            skip_write=skip_write,
            multiprocess_write=multiprocess_write,
            selected_qubits=selected_qubits,
            # estimation of given operators
            given_operators=given_operators,
            accuracy_prob_comp_delta=accuracy_prob_comp_delta,
            max_shadow_norm=max_shadow_norm,
            # other config
            rho_method=rho_method,
            trace_method=trace_method,
            estimate_trace_method=estimate_trace_method,
            counts_used=counts_used,
            **analysis_args,
        )

    #     if specific_analysis_args is None:
    #         specific_analysis_args = {}
    #     if summoner_id not in self.multimanagers:
    #         raise ValueError("No such summoner_id in multimanagers.")
    #     current_multimanager = self.multimanagers[summoner_id]

    #     counts_check = [
    #         k
    #         for k in current_multimanager.beforewards.circuits_map.keys()
    #         if len(self.exps[k].afterwards.counts) == 0
    #     ]
    #     if len(counts_check) > 0:
    #         raise ValueError(
    #             f"Counts of {len(counts_check)} experiments are empty, "
    #             + f"please check them before analysis: {counts_check}."
    #         )

    #     idx_tagmap_quantities = len(current_multimanager.quantity_container)
    #     name = (
    #         analysis_name
    #         if no_serialize
    #         else f"{analysis_name}." + f"{idx_tagmap_quantities + 1}".rjust(RJUST_LEN, "0")
    #     )

    #     all_counts_progress = qurry_progressbar(
    #         current_multimanager.beforewards.circuits_map.keys(),
    #         desc="Preparing analyzing for multiprocessing...",
    #     )

    #     quantities_input_list = []
    #     for k in all_counts_progress:
    #         if k in specific_analysis_args:
    #             v_args = specific_analysis_args[k]
    #             if isinstance(v_args, bool):
    #                 if v_args is False:
    #                     all_counts_progress.set_description_str(
    #                         f"Skipped {k} in {current_multimanager.summoner_id}."
    #                     )
    #                     continue
    #                 quantities_input_list.append(
    #                     quantities_input_collecter(
    #                         current_exps=current_multimanager.exps[k],
    #                         selected_qubits=selected_qubits,
    #                         # estimation of given operators
    #                         given_operators=given_operators,
    #                         accuracy_prob_comp_delta=accuracy_prob_comp_delta,
    #                         max_shadow_norm=max_shadow_norm,
    #                         # other config
    #                         rho_method=rho_method,
    #                         trace_method=trace_method,
    #                         estimate_trace_method=estimate_trace_method,
    #                         counts_used=counts_used,
    #                     )
    #                 )
    #             else:
    #                 quantities_input_list.append(
    #                     quantities_input_collecter(
    #                         current_exps=current_multimanager.exps[k],
    #                         selected_qubits=v_args.get("selected_qubits", selected_qubits),
    #                         # estimation of given operators
    #                         given_operators=v_args.get("given_operators", given_operators),
    #                         accuracy_prob_comp_delta=v_args.get(
    #                             "accuracy_prob_comp_delta", accuracy_prob_comp_delta
    #                         ),
    #                         max_shadow_norm=v_args.get("max_shadow_norm", max_shadow_norm),
    #                         # other config
    #                         rho_method=v_args.get("rho_method", rho_method),
    #                         trace_method=v_args.get("trace_method", trace_method),
    #                         estimate_trace_method=v_args.get(
    #                             "estimate_trace_method", estimate_trace_method
    #                         ),
    #                         counts_used=v_args.get("counts_used", counts_used),
    #                     )
    #                 )
    #         else:
    #             quantities_input_list.append(
    #                 quantities_input_collecter(
    #                     current_exps=current_multimanager.exps[k],
    #                     selected_qubits=selected_qubits,
    #                     # estimation of given operators
    #                     given_operators=given_operators,
    #                     accuracy_prob_comp_delta=accuracy_prob_comp_delta,
    #                     max_shadow_norm=max_shadow_norm,
    #                     # other config
    #                     rho_method=rho_method,
    #                     trace_method=trace_method,
    #                     estimate_trace_method=estimate_trace_method,
    #                     counts_used=counts_used,
    #                 )
    #             )

    #     current_multimanager.quantity_container[name] = TagList()
    #     pool = get_context("spawn").Pool(processes=DEFAULT_POOL_SIZE)
    #     with pool as p:
    #         outside_analyses_iterable = qurry_progressbar(
    #             p.imap_unordered(outside_analyze_wrapper, quantities_input_list),
    #             desc="Executing analysis...",
    #             total=len(quantities_input_list),
    #         )
    #         for exp_id, report in outside_analyses_iterable:
    #             current_multimanager.exps[exp_id].outside_analysis_recover(report)
    #             main, _tales = report.dumps()
    #             current_multimanager.quantity_container[name][
    #                 current_multimanager.exps[exp_id].commons.tags
    #             ].append(main)

    #     current_multimanager.multicommons.datetimes.add_only(name)

    #     if not skip_write:
    #         self.multiWrite(summoner_id=summoner_id, multiprocess_write=multiprocess_write)

    #     return current_multimanager.multicommons.summoner_id
