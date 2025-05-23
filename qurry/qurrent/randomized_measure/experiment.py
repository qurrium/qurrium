"""EntropyMeasureRandomized - Experiment (:mod:`qurry.qurrent.randomized_measure.experiment`)"""

from typing import Union, Optional, Type, Any
from collections.abc import Iterable, Hashable
import tqdm

from qiskit import QuantumCircuit

from .analysis import EntropyMeasureRandomizedAnalysis
from .arguments import EntropyMeasureRandomizedArguments, SHORT_NAME
from .utils import (
    randomized_circuit_method,
    randomized_entangled_entropy_complex,
    bitstring_mapping_getter,
)
from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...qurrium.utils.randomized import (
    random_unitary,
    local_unitary_op_to_list,
    local_unitary_op_to_pauli_coeff,
)
from ...qurrium.utils.random_unitary import check_input_for_experiment
from ...process.utils import qubit_mapper
from ...process.randomized_measure.entangled_entropy import (
    EntangledEntropyResultMitigated,
    PostProcessingBackendLabel,
    DEFAULT_PROCESS_BACKEND,
)
from ...tools import ParallelManager, set_pbar_description
from ...exceptions import RandomizedMeasureUnitaryOperatorNotFullCovering


class EntropyMeasureRandomizedExperiment(
    ExperimentPrototype[
        EntropyMeasureRandomizedArguments,
        EntropyMeasureRandomizedAnalysis,
    ]
):
    """The instance of experiment."""

    __name__ = "EntropyMeasureRandomizedExperiment"

    @property
    def arguments_instance(self) -> Type[EntropyMeasureRandomizedArguments]:
        """The arguments instance for this experiment."""
        return EntropyMeasureRandomizedArguments

    @property
    def analysis_instance(self) -> Type[EntropyMeasureRandomizedAnalysis]:
        """The analysis instance for this experiment."""
        return EntropyMeasureRandomizedAnalysis

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
    ) -> tuple[EntropyMeasureRandomizedArguments, Commonparams, dict[str, Any]]:
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
                The selected qubits for the measurement.
                If it is None, then it will return the mapping of all qubits.
                If it is int, then it will return the mapping of the last n qubits.
                If it is tuple, then it will return the mapping of the qubits in the range.
                If it is list, then it will return the mapping of the selected qubits.
                Defaults to `None`.
            unitary_loc (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator. Defaults to `None`.
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
                you can use the function :func:`generate_random_unitary_seeds`
                in :mod:`qurry.qurrium.utils.random_unitary`.

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
            raise TypeError(f"times should be an integer, but got {times}.")

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
        return EntropyMeasureRandomizedArguments._filter(
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
        arguments: EntropyMeasureRandomizedArguments,
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
                Defaults to `None`.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the side products.
        """
        side_product = {}
        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)

        set_pbar_description(pbar, f"Preparing {arguments.times} random unitary.")
        assert arguments.unitary_located is not None, "unitary_located should be specified."
        unitary_dicts = {
            n_u_i: {
                n_u_qi: (
                    random_unitary(2)
                    if arguments.random_unitary_seeds is None
                    else random_unitary(2, arguments.random_unitary_seeds[n_u_i][seed_i])
                )
                for seed_i, n_u_qi in enumerate(arguments.unitary_located)
            }
            for n_u_i in range(arguments.times)
        }

        set_pbar_description(pbar, f"Building {arguments.times} circuits.")
        assert arguments.registers_mapping is not None, "registers_mapping should be specified."
        if multiprocess:
            pool = ParallelManager()
            circ_list = pool.starmap(
                randomized_circuit_method,
                [
                    (
                        n_u_i,
                        target_circuit,
                        target_key,
                        arguments.exp_name,
                        arguments.registers_mapping,
                        unitary_dicts[n_u_i],
                    )
                    for n_u_i in range(arguments.times)
                ],
            )
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = pool.starmap(
                local_unitary_op_to_list,
                [(unitary_dicts[n_u_i],) for n_u_i in range(arguments.times)],
            )
            set_pbar_description(pbar, "Writing 'randomized'.")
            randomized_list = pool.starmap(
                local_unitary_op_to_pauli_coeff,
                [(unitary_operator_list[n_u_i],) for n_u_i in range(arguments.times)],
            )
        else:
            circ_list = [
                randomized_circuit_method(
                    n_u_i,
                    target_circuit,
                    target_key,
                    arguments.exp_name,
                    arguments.registers_mapping,
                    unitary_dicts[n_u_i],
                )
                for n_u_i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = [
                local_unitary_op_to_list(unitary_dicts[n_u_i]) for n_u_i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'randomized'.")
            randomized_list = [
                local_unitary_op_to_pauli_coeff(unitary_operator_list[n_u_i])
                for n_u_i in range(arguments.times)
            ]

        side_product["unitaryOP"] = dict(enumerate(unitary_operator_list))
        side_product["randomized"] = dict(enumerate(randomized_list))

        return circ_list, side_product

    def analyze(
        self,
        selected_qubits: Optional[Iterable[int]] = None,
        independent_all_system: bool = False,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        counts_used: Optional[Iterable[int]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> EntropyMeasureRandomizedAnalysis:
        """Calculate entangled entropy with more information combined.

        Args:
            selected_qubits (Optional[Iterable[int]], optional):
                The selected qubits. Defaults to `None`.
            independent_all_system (bool, optional):
                If True, then calculate the all system independently. Defaults to False.
            backend (PostProcessingBackendLabel, optional):
                The backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used. Defaults to `None`.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to `None`.

        Returns:
            EntropyMeasureRandomizedAnalysis: The result of the analysis.
        """
        if selected_qubits is None:
            raise ValueError("selected_qubits should be specified.")
        assert self.args.registers_mapping is not None, "registers_mapping should be not None."

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

        available_all_system_source = [
            k
            for k, v in self.reports.items()
            if (
                v.content.all_system_source == "independent"
                and v.content.counts_used == counts_used
            )
        ]
        all_system_source = (
            self.reports[available_all_system_source[-1]]
            if len(available_all_system_source) > 0 and not independent_all_system
            else None
        )

        selected_qubits = [qi % self.args.actual_num_qubits for qi in selected_qubits]
        if len(set(selected_qubits)) != len(selected_qubits):
            raise ValueError(
                f"selected_qubits should not have duplicated elements, but got {selected_qubits}."
            )
        qs = self.quantities(
            shots=self.commons.shots,
            counts=counts,
            selected_classical_registers=[final_mapping[qi] for qi in selected_qubits],
            all_system_source=all_system_source,
            backend=backend,
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
        selected_classical_registers: Optional[Iterable[int]] = None,
        all_system_source: Optional[EntropyMeasureRandomizedAnalysis] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> EntangledEntropyResultMitigated:
        """Randomized entangled entropy with complex.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The counts of the experiment.
            selected_classical_registers (Optional[Iterable[int]], optional):
                The selected classical registers. Defaults to `None`.
            all_system_source (Optional[EntropyRandomizedAnalysis], optional):
                The source of all system. Defaults to `None`.
            backend (PostProcessingBackendLabel, optional):
                The backend label. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to `None`.

        Returns:
            EntangledEntropyResultMitigated: The result of the entangled entropy.
        """

        if shots is None or counts is None:
            raise ValueError("shots and counts should be given.")

        return randomized_entangled_entropy_complex(
            shots=shots,
            counts=counts,
            selected_classical_registers=selected_classical_registers,
            all_system_source=all_system_source,
            backend=backend,
            pbar=pbar,
        )
