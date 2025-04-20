"""EchoListenRandomized - Qurry
(:mod:`qurry.qurrech.randomized_measure.qurry`)

"""

from pathlib import Path
from typing import Union, Optional, Any, Type, Literal, Iterable
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler.passmanager import PassManager

from .arguments import (
    SHORT_NAME,
    EchoListenRandomizedOutputArgs,
    EchoListenRandomizedMeasureArgs,
    EchoListenRandomizedAnalyzeArgs,
)
from .experiment import (
    EchoListenRandomizedExperiment,
    PostProcessingBackendLabel,
    DEFAULT_PROCESS_BACKEND,
)
from ...qurrium.qurrium import QurriumPrototype
from ...qurrium.utils import passmanager_processor
from ...tools.backend import GeneralSimulator
from ...declare import BaseRunArgs, TranspileArgs


class EchoListenRandomized(QurriumPrototype[EchoListenRandomizedExperiment]):
    """Randomized Measure for wave function overlap.
    a.k.a. loschmidt echo when processes time evolution system.

    Reference:
        .. note::
            - Statistical correlations between locally randomized measurements:
            A toolbox for probing entanglement in many-body quantum states -
            A. Elben, B. Vermersch, C. F. Roos, and P. Zoller,
            [PhysRevA.99.052323](
                https://doi.org/10.1103/PhysRevA.99.052323
            )

        .. code-block:: bibtex
            @article{PhysRevA.99.052323,
                title = {Statistical correlations between locally randomized measurements:
                A toolbox for probing entanglement in many-body quantum states},
                author = {Elben, A. and Vermersch, B. and Roos, C. F. and Zoller, P.},
                journal = {Phys. Rev. A},
                volume = {99},
                issue = {5},
                pages = {052323},
                numpages = {12},
                year = {2019},
                month = {May},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevA.99.052323},
                url = {https://link.aps.org/doi/10.1103/PhysRevA.99.052323}
            }
    """

    __name__ = "EchoListenRandomized"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> Type[EchoListenRandomizedExperiment]:
        """The container class responding to this Qurrium class."""
        return EchoListenRandomizedExperiment

    def measure_to_output(
        self,
        wave1: Optional[Union[QuantumCircuit, Hashable]] = None,
        wave2: Optional[Union[QuantumCircuit, Hashable]] = None,
        times: int = 100,
        measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
        measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
        unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_not_cover_measure: bool = False,
        second_backend: Optional[Backend] = None,
        second_transpile_args: Optional[TranspileArgs] = None,
        second_passmanager: Optional[Union[str, PassManager, tuple[str, PassManager]]] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager: Optional[Union[str, PassManager, tuple[str, PassManager]]] = None,
        tags: Optional[tuple[str, ...]] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> EchoListenRandomizedOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            times (int, optional):
                The number of random unitary operator.
                It will denote as `N_U` in the experiment name.
                Defaults to `100`.
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
            unitary_loc_1 (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator for the first quantum circuit.
                Defaults to `None`.
            unitary_loc_2 (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator for the second quantum circuit.
                Defaults to `None`.
            unitary_loc_not_cover_measure (bool, optional):
                Whether the range of the unitary operator is not cover the measure range.
                Defaults to False.
            second_backend (Optional[Backend], optional):
                The extra backend for the second quantum circuit.
                If None, then use the same backend as the first quantum circuit.
                Defaults to `None`.
            second_transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`
                for the second quantum circuit. Defaults to `None`.
            second_passmanager (
                Optional[Union[str, PassManager, tuple[str, PassManager]], optional
            ):
                The passmanager for the second quantum circuit. Defaults to `None`.
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
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to `None`.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                Arguments for :meth:`Backend.run`. Defaults to `None`.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to `None`.
            passmanager (Optional[Union[str, PassManager, tuple[str, PassManager]], optional):
                The passmanager. Defaults to `None`.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to `None`.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
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
            EchoListenRandomizedOutputArgs: The output arguments.
        """
        if wave1 is None:
            raise ValueError("The `wave` must be provided.")
        if wave2 is None:
            raise ValueError("The `wave2` must be provided.")

        second_passmanager_pair = passmanager_processor(
            passmanager=second_passmanager, passmanager_container=self.passmanagers
        )

        return {
            "circuits": [wave1, wave2],
            "times": times,
            "measure_1": measure_1,
            "measure_2": measure_2,
            "unitary_loc_1": unitary_loc_1,
            "unitary_loc_2": unitary_loc_2,
            "unitary_loc_not_cover_measure": unitary_loc_not_cover_measure,
            "second_backend": second_backend,
            "second_transpile_args": second_transpile_args,
            "random_unitary_seeds": random_unitary_seeds,
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
            "mode": mode,
            "indent": indent,
            "encoding": encoding,
            "jsonable": jsonable,
            "pbar": pbar,
            "second_passmanager_pair": second_passmanager_pair,
        }

    def measure(
        self,
        wave1: Optional[Union[QuantumCircuit, Hashable]] = None,
        wave2: Optional[Union[QuantumCircuit, Hashable]] = None,
        times: int = 100,
        measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
        measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
        unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
        unitary_loc_not_cover_measure: bool = False,
        second_backend: Optional[Backend] = None,
        second_transpile_args: Optional[TranspileArgs] = None,
        second_passmanager: Optional[Union[str, PassManager, tuple[str, PassManager]]] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager: Optional[Union[str, PassManager, tuple[str, PassManager]]] = None,
        tags: Optional[tuple[str, ...]] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> str:
        """Execute the experiment.

        Args:
            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            times (int, optional):
                The number of random unitary operator.
                It will denote as `N_U` in the experiment name.
                Defaults to `100`.
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
            unitary_loc_1 (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator for the first quantum circuit.
                Defaults to `None`.
            unitary_loc_2 (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator for the second quantum circuit.
                Defaults to `None`.
            unitary_loc_not_cover_measure (bool, optional):
                Whether the range of the unitary operator is not cover the measure range.
                Defaults to False.
            second_backend (Optional[Backend], optional):
                The extra backend for the second quantum circuit.
                If None, then use the same backend as the first quantum circuit.
                Defaults to `None`.
            second_transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`
                for the second quantum circuit. Defaults to `None`.
            second_passmanager (
                Optional[Union[str, PassManager, tuple[str, PassManager]], optional
            ):
                The passmanager for the second quantum circuit. Defaults to `None`.
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
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to `None`.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it
                when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                Arguments for :meth:`Backend.run`. Defaults to `None`.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to `None`.
            passmanager (Optional[Union[str, PassManager, tuple[str, PassManager]], optional):
                The passmanager. Defaults to `None`.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to `None`.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
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

        output_args = self.measure_to_output(
            wave1=wave1,
            wave2=wave2,
            times=times,
            measure_1=measure_1,
            measure_2=measure_2,
            unitary_loc_1=unitary_loc_1,
            unitary_loc_2=unitary_loc_2,
            unitary_loc_not_cover_measure=unitary_loc_not_cover_measure,
            second_backend=second_backend,
            second_transpile_args=second_transpile_args,
            second_passmanager=second_passmanager,
            random_unitary_seeds=random_unitary_seeds,
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
            mode=mode,
            indent=indent,
            encoding=encoding,
            jsonable=jsonable,
            pbar=pbar,
        )

        return self.output(**output_args)

    def multiOutput(
        self,
        config_list: list[Union[dict[str, Any], EchoListenRandomizedMeasureArgs]],
        summoner_name: str = "exps",
        summoner_id: Optional[str] = None,
        shots: int = 1024,
        backend: Backend = GeneralSimulator(),
        tags: Optional[tuple[str, ...]] = None,
        manager_run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        save_location: Union[Path, str] = Path("./"),
        skip_build_write: bool = False,
        skip_output_write: bool = False,
        multiprocess_build: bool = False,
        multiprocess_write: bool = False,
    ) -> str:
        """Output the multiple experiments.

        Args:
            config_list (list[Union[dict[str, Any], EchoListenRandomizedMeasureArgs]]):
                The list of default configurations of multiple experiment. Defaults to [].
            summoner_name (str, optional):
                Name for multimanager. Defaults to 'exps'.
            summoner_id (Optional[str], optional):
                Name for multimanager. Defaults to `None`.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The quantum backend.
                Defaults to AerSimulator().
            tags (Optional[tuple[str, ...]], optional):
                Tags of experiment of :cls:`MultiManager`. Defaults to `None`.
            manager_run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                The extra arguments for running the job,
                but for all experiments in the multimanager.
                For :meth:`backend.run()` from :cls:`qiskit.providers.backend`. Defaults to `{}`.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            skip_build_write (bool, optional):
                Whether to skip the file writing during the building.
                Defaults to False.
            skip_output_write (bool, optional):
                Whether to skip the file writing during the output.
                Defaults to False.
            multiprocess_build (bool, optional):
                Whether use multiprocess for building. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

        Returns:
            str: The summoner_id of multimanager.
        """

        return super().multiOutput(
            config_list=config_list,
            summoner_name=summoner_name,
            summoner_id=summoner_id,
            shots=shots,
            backend=backend,
            tags=tags,
            manager_run_args=manager_run_args,
            save_location=save_location,
            skip_build_write=skip_build_write,
            skip_output_write=skip_output_write,
            multiprocess_write=multiprocess_write,
            multiprocess_build=multiprocess_build,
        )

    def multiAnalysis(
        self,
        summoner_id: str,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: Optional[
            dict[Hashable, Union[dict[str, Any], EchoListenRandomizedAnalyzeArgs, bool]]
        ] = None,
        skip_write: bool = False,
        multiprocess_write: bool = False,
        # analysis arguments
        selected_classical_registers: Optional[Iterable[int]] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        counts_used: Optional[Iterable[int]] = None,
        **analysis_args,
    ) -> str:
        """Run the analysis for multiple experiments.

        Args:
            summoner_id (str): The summoner_id of multimanager.
            analysis_name (str, optional):
                The name of analysis. Defaults to 'report'.
            no_serialize (bool, optional):
                Whether to serialize the analysis. Defaults to False.
            specific_analysis_args
                Optional[dict[Hashable, Union[
                    dict[str, Any], EchoListenRandomizedAnalyzeArgs, bool]
                ]]], optional
            ):
                The specific arguments for analysis. Defaults to `None`.
            skip_write (bool, optional):
                Whether to skip the file writing during the analysis. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

            selected_classical_registers (Optional[Iterable[int]], optional):
                The list of **the index of the selected_classical_registers**.
                It's not the qubit index of first or second quantum circuit,
                but their corresponding classical registers.
                Defaults to `None`.
            backend (PostProcessingBackendLabel, optional):
                The backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used. Defaults to `None`.

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
            selected_classical_registers=selected_classical_registers,
            counts_used=counts_used,
            backend=backend,
            **analysis_args,
        )
