"""EchoListenRandomized - Experiment (:mod:`qurry.qurrech.randomized_measure.experiment`)"""

from typing import Union, Optional, Any, Type, Literal
from collections.abc import Iterable, Hashable
from pathlib import Path
import warnings
import tqdm

from qiskit import transpile, QuantumCircuit
from qiskit.providers import Backend, JobV1 as Job
from qiskit.transpiler.passmanager import PassManager

from .analysis import EchoListenRandomizedAnalysis
from .arguments import EchoListenRandomizedArguments, SHORT_NAME
from ...qurrent.randomized_measure.utils import randomized_circuit_method, bitstring_mapping_getter
from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...qurrium.experiment.utils import memory_usage_factor_expect
from ...qurrium.utils import get_counts_and_exceptions, qasm_dumps
from ...qurrium.utils.randomized import (
    random_unitary,
    local_unitary_op_to_list,
    local_unitary_op_to_pauli_coeff,
)
from ...qurrium.utils.random_unitary import check_input_for_experiment
from ...process.utils import qubit_mapper, counts_under_degree_pyrust
from ...process.availability import PostProcessingBackendLabel
from ...process.randomized_measure.wavefunction_overlap import (
    randomized_overlap_echo,
    DEFAULT_PROCESS_BACKEND,
    WaveFuctionOverlapResult,
)
from ...tools import ParallelManager, set_pbar_description, backend_name_getter
from ...declare import BaseRunArgs, TranspileArgs
from ...exceptions import (
    RandomizedMeasureUnitaryOperatorNotFullCovering,
    OverlapComparisonSizeDifferent,
    SeperatedExecutingOverlapResult,
    QurryTranspileConfigurationIgnored,
)


class EchoListenRandomizedExperiment(
    ExperimentPrototype[EchoListenRandomizedArguments, EchoListenRandomizedAnalysis]
):
    """The instance of experiment."""

    __name__ = "EchoListenRandomizedExperiment"

    @property
    def arguments_instance(self) -> Type[EchoListenRandomizedArguments]:
        """The arguments instance for this experiment."""
        return EchoListenRandomizedArguments

    @property
    def analysis_instance(self) -> Type[EchoListenRandomizedAnalysis]:
        """The analysis instance for this experiment."""
        return EchoListenRandomizedAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        times: int = 100,
        measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
        measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
        unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_not_cover_measure: bool = False,
        second_backend: Optional[Union[Backend, str]] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        **custom_kwargs: Any,
    ) -> tuple[EchoListenRandomizedArguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            times (int):
                The number of random unitary operator. Defaults to 100.
                It will denote as `N_U` in the experiment name.
            measure_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The selected qubits for the measurement for the first quantum circuit.
                If it is None, then it will return the mapping of all qubits.
                If it is int, then it will return the mapping of the last n qubits.
                If it is tuple, then it will return the mapping of the qubits in the range.
                If it is list, then it will return the mapping of the selected qubits.
                Defaults to `None`.
            measure_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The selected qubits for the measurement for the second quantum circuit.
                If it is None, then it will return the mapping of all qubits.
                If it is int, then it will return the mapping of the last n qubits.
                If it is tuple, then it will return the mapping of the qubits in the range.
                If it is list, then it will return the mapping of the selected qubits.
                Defaults to `None`.
            unitary_loc_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator for the first quantum circuit.
                Defaults to `None`.
            unitary_loc_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator for the second quantum circuit.
                Defaults to `None`.
            unitary_loc_not_cover_measure (bool, optional):
                Confirm that not all unitary operator are covered by the measure.
                If True, then close the warning.
                Defaults to False.
            second_backend (Optional[Union[Backend, str]], optional):
                The extra backend for the second quantum circuit.
                If None, then use the same backend as the first quantum circuit.
                Defaults to `None`.
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

        target_key_1, target_circuit_1 = targets[0]
        actual_qubits_1 = target_circuit_1.num_qubits
        target_key_2, target_circuit_2 = targets[1]
        actual_qubits_2 = target_circuit_2.num_qubits

        if actual_qubits_1 != actual_qubits_2:
            if any([measure_1 is None, measure_2 is None]):
                raise ValueError(
                    "When the number of qubits in two circuits is not the same, "
                    + "the measure range of two circuits should be specified."
                )
            if any([unitary_loc_1 is None, unitary_loc_2 is None]):
                raise ValueError(
                    "When the number of qubits in two circuits is not the same, "
                    + "the unitary location of two circuits should be specified."
                )

        registers_mapping_1 = qubit_mapper(actual_qubits_1, measure_1)
        qubits_measured_1 = list(registers_mapping_1)
        unitary_located_mapping_1 = qubit_mapper(actual_qubits_1, unitary_loc_1)
        assert list(unitary_located_mapping_1.values()) == list(
            range(len(unitary_located_mapping_1))
        ), "The unitary_located_mapping_1 should be continuous."
        measured_but_not_unitary_located_1 = [
            qi for qi in qubits_measured_1 if qi not in unitary_located_mapping_1
        ]

        registers_mapping_2 = qubit_mapper(actual_qubits_2, measure_2)
        qubits_measured_2 = list(registers_mapping_2)
        unitary_located_mapping_2 = qubit_mapper(actual_qubits_2, unitary_loc_2)
        assert list(unitary_located_mapping_2.values()) == list(
            range(len(unitary_located_mapping_2))
        ), "The unitary_located_mapping_2 should be continuous."
        measured_but_not_unitary_located_2 = [
            qi for qi in qubits_measured_2 if qi not in unitary_located_mapping_2
        ]

        if len(qubits_measured_1) != len(qubits_measured_2):
            raise OverlapComparisonSizeDifferent(
                "The qubits number of measuring range in two circuits should be the same, "
                + "but got different number of qubits measured."
                + f"Got circuit 1: {len(qubits_measured_1)} {qubits_measured_1}"
                + f"and circuit 2: {len(qubits_measured_2)} {qubits_measured_2}."
            )
        if len(unitary_located_mapping_1) != len(unitary_located_mapping_2):
            raise OverlapComparisonSizeDifferent(
                "The qubits number of unitary location in two circuits should be the same, "
                + "but got different number of qubits located."
                + f"Got circuit 1: {len(unitary_located_mapping_1)} {unitary_located_mapping_1}"
                + f"and circuit 2: {len(unitary_located_mapping_2)} {unitary_located_mapping_2}."
            )

        if not unitary_loc_not_cover_measure:
            if measured_but_not_unitary_located_1:
                raise RandomizedMeasureUnitaryOperatorNotFullCovering(
                    f"Some qubits {measured_but_not_unitary_located_1} are measured "
                    + "but not random unitary located in first circuit. "
                    + f"unitary_loc_1: {unitary_loc_1}, measure_1: {measure_1} "
                    + "If you are sure about this, "
                    + "you can set `unitary_loc_not_cover_measure=True` "
                    + "to close this warning.",
                )
            if measured_but_not_unitary_located_2:
                raise RandomizedMeasureUnitaryOperatorNotFullCovering(
                    f"Some qubits {measured_but_not_unitary_located_2} are measured "
                    + "but not random unitary located in second circuit. "
                    + f"unitary_loc_2: {unitary_loc_2}, measure_2: {measure_2} "
                    + "If you are sure about this, "
                    + "you can set `unitary_loc_not_cover_measure=True` "
                    + "to close this warning.",
                )

        exp_name = f"{exp_name}.N_U_{times}.{SHORT_NAME}"

        check_input_for_experiment(times, len(unitary_located_mapping_1), random_unitary_seeds)
        check_input_for_experiment(times, len(unitary_located_mapping_2), random_unitary_seeds)

        if not any([isinstance(second_backend, Backend), second_backend is None]):
            raise TypeError(
                f"second_backend should be Backend or not given, but got {type(second_backend)}."
            )

        # pylint: disable=protected-access
        return EchoListenRandomizedArguments._filter(
            exp_name=exp_name,
            target_keys=[target_key_1, target_key_2],
            times=times,
            qubits_measured_1=qubits_measured_1,
            qubits_measured_2=qubits_measured_2,
            registers_mapping_1=registers_mapping_1,
            registers_mapping_2=registers_mapping_2,
            actual_num_qubits_1=actual_qubits_1,
            actual_num_qubits_2=actual_qubits_2,
            unitary_located_mapping_1=unitary_located_mapping_1,
            unitary_located_mapping_2=unitary_located_mapping_2,
            second_backend=second_backend,
            random_unitary_seeds=random_unitary_seeds,
            **custom_kwargs,
        )
        # pylint: enable=protected-access

    @classmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: EchoListenRandomizedArguments,
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
                Defaults to `None`.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the side products.
        """
        side_product = {}
        target_key_1, target_circuit_1 = targets[0]
        target_key_1 = "" if isinstance(target_key_1, int) else str(target_key_1)
        target_key_2, target_circuit_2 = targets[1]
        target_key_2 = "" if isinstance(target_key_2, int) else str(target_key_2)

        set_pbar_description(pbar, f"Preparing {arguments.times} random unitary.")
        assert (
            arguments.unitary_located_mapping_1 is not None
        ), "unitary_located_1 should be specified."
        assert (
            arguments.unitary_located_mapping_2 is not None
        ), "unitary_located_2 should be specified."
        assert len(arguments.unitary_located_mapping_1) == len(
            arguments.unitary_located_mapping_2
        ), (
            "The number of unitary_located_mapping_1 and "
            + "unitary_located_mapping_2 should be the same, "
            + f"but got {len(arguments.unitary_located_mapping_1)} "
            + f"and {len(arguments.unitary_located_mapping_2)}."
        )
        unitary_dicts_source = {
            n_u_i: {
                ui: (
                    random_unitary(2)
                    if arguments.random_unitary_seeds is None
                    else random_unitary(2, arguments.random_unitary_seeds[n_u_i][ui])
                )
                for ui in range(len(arguments.unitary_located_mapping_1))
            }
            for n_u_i in range(arguments.times)
        }
        unitary_dict = {}
        for n_u_i in range(arguments.times):
            unitary_dict[n_u_i] = {
                qi: unitary_dicts_source[n_u_i][ui]
                for qi, ui in arguments.unitary_located_mapping_1.items()
            }
            unitary_dict[n_u_i + arguments.times] = {
                qi: unitary_dicts_source[n_u_i][ui]
                for qi, ui in arguments.unitary_located_mapping_2.items()
            }

        set_pbar_description(pbar, f"Building {arguments.times * 2} circuits.")
        assert arguments.registers_mapping_1 is not None, "registers_mapping_1 should be specified."
        assert arguments.registers_mapping_2 is not None, "registers_mapping_2 should be specified."
        if multiprocess:
            pool = ParallelManager()
            circ_list = pool.starmap(
                randomized_circuit_method,
                [
                    (
                        n_u_i,
                        target_circuit_1,
                        target_key_1,
                        arguments.exp_name,
                        arguments.registers_mapping_1,
                        unitary_dict[n_u_i],
                    )
                    for n_u_i in range(arguments.times)
                ]
                + [
                    (
                        n_u_i + arguments.times,
                        target_circuit_2,
                        target_key_2,
                        arguments.exp_name,
                        arguments.registers_mapping_2,
                        unitary_dict[n_u_i + arguments.times],
                    )
                    for n_u_i in range(arguments.times)
                ],
            )
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = pool.starmap(
                local_unitary_op_to_list,
                [(unitary_dicts_source[n_u_i],) for n_u_i in range(arguments.times)],
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
                    target_circuit_1,
                    target_key_1,
                    arguments.exp_name,
                    arguments.registers_mapping_1,
                    unitary_dict[n_u_i],
                )
                for n_u_i in range(arguments.times)
            ] + [
                randomized_circuit_method(
                    n_u_i + arguments.times,
                    target_circuit_2,
                    target_key_2,
                    arguments.exp_name,
                    arguments.registers_mapping_2,
                    unitary_dict[n_u_i + arguments.times],
                )
                for n_u_i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'unitaryOP'.")
            unitary_operator_list = [
                local_unitary_op_to_list(unitary_dicts_source[n_u_i])
                for n_u_i in range(arguments.times)
            ]
            set_pbar_description(pbar, "Writing 'randomized'.")
            randomized_list = [
                local_unitary_op_to_pauli_coeff(unitary_operator_list[n_u_i])
                for n_u_i in range(arguments.times)
            ]
        assert len(circ_list) == 2 * arguments.times, "The number of circuits is not correct."

        side_product["unitaryOP"] = dict(enumerate(unitary_operator_list))
        side_product["randomized"] = dict(enumerate(randomized_list))

        return circ_list, side_product

    @classmethod
    def build(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager_pair: Optional[tuple[str, PassManager]] = None,
        tags: Optional[tuple[str, ...]] = None,
        # multimanager
        default_analysis: Optional[list[dict[str, Any]]] = None,
        serial: Optional[int] = None,
        summoner_id: Optional[Hashable] = None,
        summoner_name: Optional[str] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
        # special
        second_passmanager_pair: Optional[tuple[str, PassManager]] = None,
        **custom_and_main_kwargs: Any,
    ):
        """Construct the experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                Arguments for :meth:`Backend.run`. Defaults to `None`.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to `None`.
            passmanager_pair (Optional[tuple[str, PassManager]], optional):
                The passmanager pair for transpile. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Given the experiment multiple tags to make a dictionary for recongnizing it.
                Defaults to None.

            default_analysis (list[dict[str, Any]], optional):
                The analysis methods will be excuted after counts has been computed.
                Defaults to [].
            serial (Optional[int], optional):
                Index of experiment in a multiOutput.
                **!!ATTENTION, this should only be used by `Multimanager`!!**
                Defaults to None.
            summoner_id (Optional[Hashable], optional):
                ID of experiment of :cls:`MultiManager`.
                **!!ATTENTION, this should only be used by `Multimanager`!!**
                Defaults to None.
            summoner_name (Optional[str], optional):
                Name of experiment of :cls:`MultiManager`.
                **!!ATTENTION, this should only be used by `Multimanager`!!**
                _description_. Defaults to None.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The export version of OpenQASM. Defaults to 'qasm3'.
            export (bool, optional):
                Whether to export the experiment. Defaults to False.
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to None.
            mode (str, optional):
                The mode to open the file. Defaults to 'w+'.
            indent (int, optional):
                The indent of json file. Defaults to 2.
            encoding (str, optional):
                The encoding of json file. Defaults to 'utf-8'.
            jsonable (bool, optional):
                Whether to jsonablize the experiment output. Defaults to False.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

            second_passmanager_pair (Optional[tuple[str, PassManager]], optional):
                The passmanager pair for transpile of the second circuit.
                Defaults to None.
            custom_and_main_kwargs (Any):
                Other custom arguments.

        Returns:
            ExperimentPrototype: The experiment.
        """

        # preparing
        set_pbar_description(pbar, "Parameter loading...")

        current_exp = cls._params_control_core(
            targets=targets,
            shots=shots,
            backend=backend,
            run_args=run_args,
            transpile_args=transpile_args,
            tags=tags,
            exp_name=exp_name,
            default_analysis=default_analysis,
            serial=serial,
            summoner_id=summoner_id,
            summoner_name=summoner_name,
            pbar=pbar,
            **custom_and_main_kwargs,
        )
        if not isinstance(current_exp.commons.backend, Backend):
            if isinstance(backend, Backend):
                set_pbar_description(pbar, "Backend replacing...")
                current_exp.replace_backend(backend)
            else:
                raise ValueError(
                    "No vaild backend to run, exisited backend: "
                    + f"{current_exp.commons.backend} as type "
                    + f"{type(current_exp.commons.backend)}, "
                    + f"given backend: {backend} as type {type(backend)}."
                )
        assert isinstance(current_exp.commons.backend, Backend), (
            f"Invalid backend: {current_exp.commons.backend} as "
            + f"type {type(current_exp.commons.backend)}."
        )

        # circuit
        set_pbar_description(pbar, "Circuit creating...")
        current_exp.beforewards.target.extend(targets)
        cirqs, side_prodict = current_exp.method(
            targets=targets, arguments=current_exp.args, pbar=pbar, multiprocess=multiprocess
        )
        current_exp.beforewards.side_product.update(side_prodict)

        # qasm
        set_pbar_description(pbar, "Exporting OpenQASM string...")
        targets_keys, targets_values = zip(*targets)
        targets_keys: tuple[Hashable, ...]
        targets_values: tuple[QuantumCircuit, ...]

        if multiprocess:
            pool = ParallelManager()
            current_exp.beforewards.circuit_qasm.extend(
                pool.starmap(qasm_dumps, ((q, qasm_version) for q in cirqs))
            )
            current_exp.beforewards.target_qasm.extend(
                zip(
                    (str(k) for k in targets_keys),
                    pool.starmap(qasm_dumps, [(q, qasm_version) for q in targets_values]),
                )
            )
        else:
            current_exp.beforewards.circuit_qasm.extend(
                (qasm_dumps(q, qasm_version) for q in cirqs)
            )
            current_exp.beforewards.target_qasm.extend(
                zip(
                    (str(k) for k in targets_keys),
                    (qasm_dumps(q, qasm_version) for q in targets_values),
                )
            )

        transpiled_circs: list[QuantumCircuit] = []
        # transpile
        if passmanager_pair is not None:
            passmanager_name, passmanager = passmanager_pair
            set_pbar_description(
                pbar, f"Circuit transpiling by passmanager '{passmanager_name}'..."
            )
            transpiled_circs += passmanager.run(circuits=cirqs[: current_exp.args.times])
            if len(current_exp.commons.transpile_args) > 0:
                warnings.warn(
                    f"Passmanager '{passmanager_name}' is given, "
                    + f"the transpile_args will be ignored in '{current_exp.exp_id}'",
                    category=QurryTranspileConfigurationIgnored,
                )
        else:
            set_pbar_description(pbar, "Circuit transpiling...")
            transpile_args = current_exp.commons.transpile_args.copy()
            transpile_args.pop("num_processes", None)
            transpiled_circs += transpile(
                cirqs[: current_exp.args.times],
                backend=current_exp.commons.backend,
                num_processes=None if multiprocess else 1,
                **transpile_args,  # type: ignore
            )

        assert isinstance(current_exp.args.second_backend, (Backend, type(None))), (
            "second_backend should be Backend or not given, "
            + f"but got {type(current_exp.args.second_backend)}."
        )
        if second_passmanager_pair is not None:
            second_passmanager_name, second_passmanager = second_passmanager_pair
            set_pbar_description(
                pbar, f"Circuit transpiling by second passmanager '{second_passmanager_name}'..."
            )
            transpiled_circs += second_passmanager.run(
                circuits=cirqs[current_exp.args.times :],
                num_processes=None if multiprocess else 1,  # type: ignore
            )
            if current_exp.args.second_transpile_args is not None:
                warnings.warn(
                    f"Passmanager '{passmanager_name}' is given, "
                    + f"the second_transpile_args will be ignored in '{current_exp.exp_id}'",
                    category=QurryTranspileConfigurationIgnored,
                )
        elif current_exp.args.second_transpile_args is not None:
            second_transpile_args = current_exp.args.second_transpile_args.copy()
            second_transpile_args.pop("num_processes", None)
            transpiled_circs += transpile(
                cirqs[current_exp.args.times :],
                backend=(
                    current_exp.commons.backend
                    if current_exp.args.second_backend is None
                    else current_exp.args.second_backend
                ),
                num_processes=None if multiprocess else 1,
                **second_transpile_args,
            )
        elif passmanager_pair is not None:
            passmanager_name, passmanager = passmanager_pair
            set_pbar_description(
                pbar, f"Circuit transpiling by passmanager '{passmanager_name}'..."
            )

            transpiled_circs += passmanager.run(
                circuits=cirqs[current_exp.args.times :],
                num_processes=None if multiprocess else 1,  # type: ignore
            )
            if len(current_exp.commons.transpile_args) > 0:
                warnings.warn(
                    f"Passmanager '{passmanager_name}' is given, "
                    + f"the transpile_args will be ignored in '{current_exp.exp_id}'",
                    category=QurryTranspileConfigurationIgnored,
                )
        else:
            set_pbar_description(pbar, "Circuit transpiling...")
            transpile_args = current_exp.commons.transpile_args.copy()
            transpile_args.pop("num_processes", None)
            transpiled_circs += transpile(
                cirqs[current_exp.args.times :],
                backend=(
                    current_exp.commons.backend
                    if current_exp.args.second_backend is None
                    else current_exp.args.second_backend
                ),
                num_processes=None if multiprocess else 1,
                **transpile_args,
            )

        assert len(transpiled_circs) == 2 * current_exp.args.times, (
            "The number of transpiled circuits is not correct, "
            + f"expected {2 * current_exp.args.times}, but got {len(transpiled_circs)}."
        )

        set_pbar_description(pbar, "Circuit loading...")
        current_exp.beforewards.circuit.extend(transpiled_circs)

        # memory usage factor
        current_exp.memory_usage_factor = memory_usage_factor_expect(
            target=current_exp.beforewards.target,
            circuits=current_exp.beforewards.circuit,
            commonparams=current_exp.commons,
        )

        # commons
        note_and_date = current_exp.commons.datetimes.add_only("build")
        set_pbar_description(
            pbar, f"Building Completed, denoted '{note_and_date[0]}' date: {note_and_date[1]}..."
        )

        # export may be slow, consider export at finish or something
        if isinstance(save_location, (Path, str)) and export:
            set_pbar_description(pbar, "Setup data exporting...")
            current_exp.write(
                save_location=save_location,
                mode=mode,
                indent=indent,
                encoding=encoding,
                jsonable=jsonable,
            )

        return current_exp

    # local execution
    def run(
        self,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> str:
        """Export the result after running the job.

        Args:
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to `None`.

        Raises:
            ValueError: No circuit ready.
            ValueError: The circuit has not been constructed yet.

        Returns:
            str: The ID of the experiment.
        """
        if len(self.beforewards.circuit) == 0:
            raise ValueError("The circuit has not been constructed yet.")

        assert isinstance(self.commons.backend, Backend), (
            f"Current backend {self.commons.backend} needs to be backend not "
            + f"{type({self.commons.backend})}."
        )
        assert hasattr(self.commons.backend, "run"), "Current backend is not runnable."

        if self.args.second_backend is None:
            set_pbar_description(pbar, "Executing with single backend...")
            event_name, date = self.commons.datetimes.add_serial("run")
            execution_1: Job = self.commons.backend.run(  # type: ignore
                self.beforewards.circuit,
                shots=self.commons.shots,
                **self.commons.run_args,
            )
            # commons
            set_pbar_description(
                pbar, f"Executing completed '{event_name}', denoted date: {date}..."
            )
            # beforewards
            self.beforewards.job_id.append(execution_1.job_id())
            # afterwards
            result_1 = execution_1.result()
            self.afterwards.result.append(result_1)

        elif not isinstance(self.args.second_backend, Backend):
            raise ValueError(
                "second_backend should be Backend or not given, "
                + f"but got {type(self.args.second_backend)}."
            )

        elif not hasattr(self.args.second_backend, "run"):
            raise ValueError("second_backend is not runnable.")

        else:
            if backend_name_getter(self.args.second_backend) == backend_name_getter(
                self.commons.backend
            ):
                warnings.warn(
                    f"The second backend {self.args.second_backend} is seem to be "
                    + f"the same as the first backend {self.commons.backend}. "
                    + "But since they will excute separately, "
                    + "it will return different results although the same backend",
                    category=SeperatedExecutingOverlapResult,
                )

            set_pbar_description(pbar, "Executing with two backends...")
            event_name, date = self.commons.datetimes.add_serial("run")
            execution_1: Job = self.commons.backend.run(  # type: ignore
                self.beforewards.circuit[: self.args.times],
                shots=self.commons.shots,
                **self.commons.run_args,
            )
            execution_2: Job = self.args.second_backend.run(  # type: ignore
                self.beforewards.circuit[self.args.times :],
                shots=self.commons.shots,
                **self.commons.run_args,
            )
            # commons
            set_pbar_description(
                pbar, f"Executing completed '{event_name}', denoted date: {date}..."
            )
            # beforewards
            self.beforewards.job_id.append(f"{execution_1.job_id()}_{execution_2.job_id()}")
            # afterwards
            result_1 = execution_1.result()
            self.afterwards.result.append(result_1)
            result_2 = execution_2.result()
            self.afterwards.result.append(result_2)

        return self.exp_id

    def result(
        self,
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> str:
        """Export the result of the experiment.

        Args:
            export (bool, optional):
                Whether to export the experiment. Defaults to False.
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to `None`.
            mode (str, optional):
                The mode to open the file. Defaults to 'w+'.
            indent (int, optional):
                The indent of json file. Defaults to 2.
            encoding (str, optional):
                The encoding of json file. Defaults to 'utf-8'.
            jsonable (bool, optional):
                Whether to jsonablize the experiment output. Defaults to False.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to `None`.

        Returns:
            str: The ID of the experiment.
        """

        if len(self.afterwards.result) == 1:
            set_pbar_description(pbar, "Result loading from single job...")
            counts_1, exceptions_1 = get_counts_and_exceptions(
                result=self.afterwards.result[0], num=self.args.times * 2
            )
            if len(exceptions_1) > 0:
                if "exceptions" not in self.outfields:
                    self.outfields["exceptions"] = {}
                for result_id, exception_item in exceptions_1.items():
                    self.outfields["exceptions"][result_id] = exception_item

            set_pbar_description(pbar, "Counts loading from single job...")
            for _c in counts_1:
                self.afterwards.counts.append(_c)

        elif len(self.afterwards.result) == 2:
            set_pbar_description(pbar, "Result loading from two jobs...")
            counts_1, exceptions_1 = get_counts_and_exceptions(
                result=self.afterwards.result[0],
                num=self.args.times,
            )
            counts_2, exceptions_2 = get_counts_and_exceptions(
                result=self.afterwards.result[1],
                num=self.args.times,
            )
            exceptions = {**exceptions_1, **exceptions_2}
            if len(exceptions) > 0:
                if "exceptions" not in self.outfields:
                    self.outfields["exceptions"] = {}
                for result_id, exception_item in exceptions.items():
                    self.outfields["exceptions"][result_id] = exception_item

            set_pbar_description(pbar, "Counts loading from two jobs...")
            for _c in counts_1 + counts_2:
                self.afterwards.counts.append(_c)

        else:
            raise ValueError(
                "The number of results should be one or two, "
                + f"but got {len(self.afterwards.result)}."
            )

        if len(self.commons.default_analysis) > 0:
            for i, _analysis in enumerate(self.commons.default_analysis):
                set_pbar_description(
                    pbar, f"Default Analysis executing {i}/{len(self.commons.default_analysis)}..."
                )
                self.analyze(**_analysis)

        if export:
            # export may be slow, consider export at finish or something
            if isinstance(save_location, (Path, str)):
                set_pbar_description(pbar, "Setup data exporting...")
                self.write(
                    save_location=save_location,
                    mode=mode,
                    indent=indent,
                    encoding=encoding,
                    jsonable=jsonable,
                )

        return self.exp_id

    def analyze(
        self,
        selected_classical_registers: Optional[Iterable[int]] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        counts_used: Optional[Iterable[int]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> EchoListenRandomizedAnalysis:
        """Calculate wave function overlap with more information combined.

        Args:
            selected_classical_registers (Optional[Iterable[int]], optional):
                The list of **the index of the selected_classical_registers**.
                It's not the qubit index of first or second quantum circuit,
                but their corresponding classical registers.
                Defaults to `None`.
            backend (PostProcessingBackendLabel, optional):
                The backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used. Defaults to `None`.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar API, you can use put a :cls:`tqdm` object here.
                This function will update the progress bar description.
                Defaults to `None`.

        Returns:
            EchoListenRandomizedAnalysis: The result of the experiment
        """

        shots = self.commons.shots
        assert self.args.registers_mapping_1 is not None, "registers_mapping_1 should be not None."
        assert self.args.registers_mapping_2 is not None, "registers_mapping_2 should be not None."
        existed_classical_registers = list(self.args.registers_mapping_1.values())
        existed_classical_registers_check = list(self.args.registers_mapping_2.values())
        assert existed_classical_registers == existed_classical_registers_check, (
            "The classical registers of two circuits should be the same, "
            + f"but got {existed_classical_registers} and {existed_classical_registers_check}."
            + f"from registers_mapping_1: {self.args.registers_mapping_1} and "
            + f"registers_mapping_2: {self.args.registers_mapping_2}."
        )
        assert existed_classical_registers == list(range(len(existed_classical_registers))), (
            "The classical registers should be continuous, "
            + f"but got {existed_classical_registers}."
        )

        classical_registers_num = len(existed_classical_registers)
        selected_classical_registers = (
            list(self.args.registers_mapping_1.values())
            if selected_classical_registers is None
            else [ci % classical_registers_num for ci in selected_classical_registers]
        )
        assert len(set(selected_classical_registers)) == len(selected_classical_registers), (
            "The selected_classical_registers should not have duplicated elements, "
            + f"but got {selected_classical_registers}."
        )
        not_existed_classical_registers = [
            ci for ci in selected_classical_registers if ci not in existed_classical_registers
        ]
        if not_existed_classical_registers:
            raise ValueError(
                f"Some classical registers {not_existed_classical_registers} "
                + "are not existed in the register mapping of two circuit. "
                + f"registers_mapping_1: {self.args.registers_mapping_1}, "
                + f"registers_mapping_2: {self.args.registers_mapping_2}, "
                + f"selected: {selected_classical_registers}"
            )

        first_countses = self.afterwards.counts[: self.args.times]
        second_countses = self.afterwards.counts[self.args.times :]
        assert len(first_countses) == len(second_countses), (
            "The number of first and second counts should be the same, "
            + f"but got {len(first_countses)} and {len(second_countses)}. "
            + f"from counts with length {len(self.afterwards.counts)}, "
            + f"times: {self.args.times}."
        )

        bitstring_mapping_1, final_mapping_1 = bitstring_mapping_getter(
            first_countses, self.args.registers_mapping_1
        )
        bitstring_mapping_2, final_mapping_2 = bitstring_mapping_getter(
            second_countses, self.args.registers_mapping_2
        )

        actual_bitstring_1_num_and_list = (
            len(list(first_countses[0].keys())[0]),
            list(final_mapping_1.values()),
        )
        first_counts_of_last_clreg = [
            counts_under_degree_pyrust(
                counts,
                actual_bitstring_1_num_and_list[0],
                actual_bitstring_1_num_and_list[1],
            )
            for counts in first_countses
        ]

        actual_bitstring_2_num_and_list = (
            len(list(second_countses[0].keys())[0]),
            list(final_mapping_2.values()),
        )
        second_counts_of_last_clreg = [
            counts_under_degree_pyrust(
                counts,
                actual_bitstring_2_num_and_list[0],
                actual_bitstring_2_num_and_list[1],
            )
            for counts in second_countses
        ]

        qs = self.quantities(
            shots=shots,
            first_counts=first_counts_of_last_clreg,
            second_counts=second_counts_of_last_clreg,
            selected_classical_registers=selected_classical_registers,
            backend=backend,
            pbar=pbar,
        )

        serial = len(self.reports)
        analysis = self.analysis_instance(
            serial=serial,
            shots=shots,
            registers_mapping_1=self.args.registers_mapping_1,
            registers_mapping_2=self.args.registers_mapping_2,
            unitary_located_mapping_1=self.args.unitary_located_mapping_1,
            unitary_located_mapping_2=self.args.unitary_located_mapping_2,
            bitstring_mapping_1=bitstring_mapping_1,
            bitstring_mapping_2=bitstring_mapping_2,
            counts_used=counts_used,
            **qs,  # type: ignore
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        first_counts: Optional[list[dict[str, int]]] = None,
        second_counts: Optional[list[dict[str, int]]] = None,
        selected_classical_registers: Optional[Iterable[int]] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> WaveFuctionOverlapResult:
        """Calculate entangled entropy with more information combined.

        Args:
            shots (int):
                Shots of the experiment on quantum machine.
            first_counts (list[dict[str, int]]):
                Counts of the experiment on quantum machine.
            second_counts (list[dict[str, int]]):
                Counts of the experiment on quantum machine.
            selected_classical_registers (Optional[Iterable[int]], optional):
                The list of **the index of the selected_classical_registers**.
            backend (ExistingProcessBackendLabel, optional):
                Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar API, you can use put a :cls:`tqdm` object here.
                This function will update the progress bar description.
                Defaults to `None`.

        Returns:
            WaveFuctionOverlapResult: A dictionary contains purity, entropy,
                a list of each overlap, puritySD, degree, actual measure range, bitstring range.
        """
        if first_counts is None or second_counts is None or shots is None:
            raise ValueError("first_counts, second_counts, and shots must be specified.")

        return randomized_overlap_echo(
            shots=shots,
            first_counts=first_counts,
            second_counts=second_counts,
            selected_classical_registers=selected_classical_registers,
            backend=backend,
            pbar=pbar,
        )
