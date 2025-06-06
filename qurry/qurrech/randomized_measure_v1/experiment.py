"""EchoListenRandomizedV1 - Experiment (:mod:`qurry.qurrech.randomized_measure_v1.experiment`)

The deprecated version of the randomized measure experiment.
"""

from typing import Union, Optional, Type, Any
from collections.abc import Iterable, Hashable
import warnings
import tqdm

from qiskit import QuantumCircuit

from .analysis import EchoListenRandomizedV1Analysis
from .arguments import EchoListenRandomizedV1Arguments, SHORT_NAME
from ...qurrent.randomized_measure_v1.utils import circuit_method_core_v1
from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...process.utils import qubit_selector
from ...qurrium.utils.randomized import (
    local_random_unitary_operators,
    local_random_unitary_pauli_coeff,
    random_unitary,
)
from ...qurrium.utils.random_unitary import check_input_for_experiment
from ...process.randomized_measure.wavefunction_overlap_v1 import (
    randomized_overlap_echo_v1,
    DEFAULT_PROCESS_BACKEND,
)
from ...process.availability import PostProcessingBackendLabel
from ...tools import qurry_progressbar, ParallelManager, set_pbar_description
from ...exceptions import QurryArgumentsExpectedNotNone


class EchoListenRandomizedV1Experiment(
    ExperimentPrototype[EchoListenRandomizedV1Arguments, EchoListenRandomizedV1Analysis]
):
    """The instance of experiment."""

    __name__ = "EchoListenRandomizedV1Experiment"

    @property
    def arguments_instance(self) -> Type[EchoListenRandomizedV1Arguments]:
        """The arguments instance for this experiment."""
        return EchoListenRandomizedV1Arguments

    @property
    def analysis_instance(self) -> Type[EchoListenRandomizedV1Analysis]:
        """The analysis instance for this experiment."""
        return EchoListenRandomizedV1Analysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        times: int = 100,
        measure: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc: Optional[Union[tuple[int, int], int]] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        **custom_kwargs: Any,
    ) -> tuple[EchoListenRandomizedV1Arguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            times (int):
                The number of random unitary operator. Defaults to 100.
                It will denote as `N_U` in the experiment name.
            measure (Optional[Union[tuple[int, int], int]]):
                The measure range. Defaults to None.
            unitary_loc (Optional[Union[tuple[int, int], int]]):
                The range of the unitary operator. Defaults to None.
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
            ValueError: If the number of target circuits is not two.
            TypeError: If times is not an integer.
            ValueError: If the number of qubits in two circuits is not the same.

        Returns:
            tuple[EntropyMeasureRandomizedArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) != 2:
            raise ValueError("The number of target circuits should be two.")
        if not isinstance(times, int):
            raise TypeError(f"times should be an integer, but got {times}.")

        target_key_01, target_circuit_01 = targets[0]
        num_qubits_01 = target_circuit_01.num_qubits
        target_key_02, target_circuit_02 = targets[1]
        num_qubits_02 = target_circuit_02.num_qubits

        if num_qubits_01 != num_qubits_02:
            raise ValueError(
                "The number of qubits in two circuits should be the same, "
                + f"but got {target_key_01}: {num_qubits_01} and {target_key_02}: {num_qubits_02}."
            )

        if measure is None:
            measure = num_qubits_01
        measure = qubit_selector(num_qubits_01, degree=measure)
        if unitary_loc is None:
            unitary_loc = num_qubits_01
        unitary_loc = qubit_selector(num_qubits_01, degree=unitary_loc)

        exp_name = f"{exp_name}.N_U_{times}.{SHORT_NAME}"

        check_input_for_experiment(times, num_qubits_01, random_unitary_seeds)

        # pylint: disable=protected-access
        return EchoListenRandomizedV1Arguments._filter(
            exp_name=exp_name,
            target_keys=[target_key_01, target_key_02],
            times=times,
            measure=measure,
            unitary_loc=unitary_loc,
            random_unitary_seeds=random_unitary_seeds,
            **custom_kwargs,
        )
        # pylint: enable=protected-access

    @classmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: EchoListenRandomizedV1Arguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (EchoListenRandomizedArguments):
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

        set_pbar_description(
            pbar,
            f"Preparing {arguments.times} random unitary with {arguments.workers_num} workers.",
        )

        target_key_01, target_circuit_01 = targets[0]
        target_key_01 = "" if isinstance(target_key_01, int) else str(target_key_01)
        num_qubits_01 = target_circuit_01.num_qubits
        target_key_02, target_circuit_02 = targets[1]
        target_key_02 = "" if isinstance(target_key_02, int) else str(target_key_02)
        num_qubits_02 = target_circuit_02.num_qubits

        assert (
            num_qubits_01 == num_qubits_02
        ), "The number of qubits in two circuits should be the same."

        if arguments.unitary_loc is None:
            actual_unitary_loc = (0, num_qubits_01)
            warnings.warn(
                f"| unitary_loc is not specified, using the whole qubits {actual_unitary_loc},"
                + " but it should be not None anymore here.",
                QurryArgumentsExpectedNotNone,
            )
        else:
            actual_unitary_loc = arguments.unitary_loc
        unitary_dict = {
            i: {
                j: (
                    random_unitary(2)
                    if arguments.random_unitary_seeds is None
                    else random_unitary(2, arguments.random_unitary_seeds[i][j])
                )
                for j in range(*actual_unitary_loc)
            }
            for i in range(arguments.times)
        }

        set_pbar_description(pbar, f"Building {arguments.times * 2} circuits.")
        assert arguments.unitary_loc is not None, "unitary_loc should be not None."
        assert arguments.measure is not None, "measure should be not None."
        if multiprocess:
            pool = ParallelManager(arguments.workers_num)
            circ_list = pool.starmap(
                circuit_method_core_v1,
                [
                    (
                        i,
                        target_circuit_01,
                        target_key_01,
                        arguments.exp_name,
                        arguments.unitary_loc,
                        unitary_dict[i],
                        arguments.measure,
                    )
                    for i in range(arguments.times)
                ]
                + [
                    (
                        i + arguments.times,
                        target_circuit_02,
                        target_key_02,
                        arguments.exp_name,
                        arguments.unitary_loc,
                        unitary_dict[i],
                        arguments.measure,
                    )
                    for i in range(arguments.times)
                ],
            )
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = pool.starmap(
                local_random_unitary_operators,
                [(arguments.unitary_loc, unitary_dict[i]) for i in range(arguments.times)],
            )
            set_pbar_description(pbar, "Writing 'randomized'.")
            randomized_list = pool.starmap(
                local_random_unitary_pauli_coeff,
                [(arguments.unitary_loc, unitary_operator_list[i]) for i in range(arguments.times)],
            )

        else:
            circ_list = [
                circuit_method_core_v1(
                    i,
                    target_circuit_01,
                    target_key_01,
                    arguments.exp_name,
                    arguments.unitary_loc,
                    unitary_dict[i],
                    arguments.measure,
                )
                for i in range(arguments.times)
            ] + [
                circuit_method_core_v1(
                    i + arguments.times,
                    target_circuit_02,
                    target_key_02,
                    arguments.exp_name,
                    arguments.unitary_loc,
                    unitary_dict[i],
                    arguments.measure,
                )
                for i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = [
                local_random_unitary_operators(arguments.unitary_loc, unitary_dict[i])
                for i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'randomized'.")
            randomized_list = [
                local_random_unitary_pauli_coeff(arguments.unitary_loc, unitary_operator_list[i])
                for i in range(arguments.times)
            ]

        assert len(circ_list) == 2 * arguments.times, "The number of circuits is not correct."

        side_product["unitaryOP"] = dict(enumerate(unitary_operator_list))
        side_product["randomized"] = dict(enumerate(randomized_list))

        return circ_list, side_product

    def analyze(
        self,
        degree: Optional[Union[tuple[int, int], int]] = None,
        counts_used: Optional[Iterable[int]] = None,
        workers_num: Optional[int] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> EchoListenRandomizedV1Analysis:
        """Calculate entangled entropy with more information combined.

        Args:
            degree (Union[tuple[int, int], int]): Degree of the subsystem.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used.
                If not specified, then use all counts.
                Defaults to None.
            workers_num (Optional[int], optional):
                Number of multi-processing workers,
                if sets to 1, then disable to using multi-processing;
                if not specified, then use the number of all cpu counts - 2 by `cpu_count() - 2`.
                Defaults to None.
            backend (PostProcessingBackendLabel, optional):
                Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional): Progress bar. Defaults to None.

        Returns:
            dict[str, float]: A dictionary contains
                purity, entropy, a list of each overlap, puritySD,
                purity of all system, entropy of all system,
                a list of each overlap in all system, puritySD of all system,
                degree, actual measure range, actual measure range in all system, bitstring range.
        """

        if degree is None:
            raise ValueError("degree must be specified, but get None.")

        len_counts = len(self.afterwards.counts)
        assert len_counts % 2 == 0, "The counts should be even."
        len_counts_half = int(len_counts / 2)
        if isinstance(counts_used, Iterable):
            if max(counts_used) >= len_counts_half:
                raise ValueError(
                    "counts_used should be less than "
                    f"{len_counts_half}, but get {max(counts_used)}."
                )
            counts = [self.afterwards.counts[i] for i in counts_used] + [
                self.afterwards.counts[i + len_counts_half] for i in counts_used
            ]
        else:
            if counts_used is not None:
                raise ValueError(f"counts_used should be Iterable, but get {type(counts_used)}.")
            counts = self.afterwards.counts

        if isinstance(pbar, tqdm.tqdm):
            qs = self.quantities(
                shots=self.commons.shots,
                counts=counts,
                degree=degree,
                measure=self.args.measure,
                backend=backend,
                workers_num=workers_num,
                pbar=pbar,
            )

        else:
            pbar_selfhost = qurry_progressbar(
                range(1),
                bar_format="simple",
            )

            with pbar_selfhost as pb_self:
                qs = self.quantities(
                    shots=self.commons.shots,
                    counts=counts,
                    degree=degree,
                    measure=self.args.measure,
                    backend=backend,
                    workers_num=workers_num,
                    pbar=pb_self,
                )
                pb_self.update()

        serial = len(self.reports)
        analysis = self.analysis_instance(
            serial=serial,
            shots=self.commons.shots,
            unitary_loc=self.args.unitary_loc,
            counts_used=counts_used,
            **qs,  # type: ignore
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        counts: Optional[list[dict[str, int]]] = None,
        degree: Optional[Union[tuple[int, int], int]] = None,
        measure: Optional[tuple[int, int]] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        workers_num: Optional[int] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> dict[str, float]:
        """Calculate entangled entropy with more information combined.

        Args:
            shots (int): Shots of the experiment on quantum machine.
            counts (list[dict[str, int]]): Counts of the experiment on quantum machine.
            degree (Union[tuple[int, int], int]): Degree of the subsystem.
            measure (tuple[int, int], optional):
                Measuring range on quantum circuits. Defaults to None.
            backend (PostProcessingBackendLabel, optional):
                Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            workers_num (Optional[int], optional):
                Number of multi-processing workers,
                if sets to 1, then disable to using multi-processing;
                if not specified, then use the number of all cpu counts - 2 by `cpu_count() - 2`.
                Defaults to None.
            pbar (Optional[tqdm.tqdm], optional): Progress bar. Defaults to None.

        Returns:
            dict[str, float]: A dictionary contains
                purity, entropy, a list of each overlap, puritySD,
                purity of all system, entropy of all system,
                a list of each overlap in all system, puritySD of all system,
                degree, actual measure range, actual measure range in all system, bitstring range.
        """
        if shots is None or counts is None:
            raise ValueError("shots and counts should be specified.")

        return randomized_overlap_echo_v1(
            shots=shots,
            counts=counts,
            degree=degree,
            measure=measure,
            backend=backend,
            workers_num=workers_num,
            pbar=pbar,
        )
