"""Qurrium - A Qiskit Macro (:mod:`qurry.qurrium.qurrium`)"""

import warnings
from abc import abstractmethod, ABC
from typing import Literal, Union, Optional, Any, Type, Generic
from collections.abc import Hashable
from pathlib import Path
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler.passmanager import PassManager

from .runner import RemoteAccessor, retrieve_counter
from .utils import passmanager_processor
from .container import (
    WaveContainer,
    ExperimentContainer,
    MultiManagerContainer,
    PassManagerContainer,
    ExperimentContainerWrapper,
    _E,
)
from .multimanager.multimanager import (
    MultiManager,
    PendingTargetProviderLiteral,
    PendingStrategyLiteral,
)
from ..tools import qurry_progressbar
from ..tools.backend import GeneralSimulator
from ..declare import BaseRunArgs, TranspileArgs, OutputArgs, BasicArgs, AnalyzeArgs


class QurriumPrototype(ABC, Generic[_E]):
    """Qurrium, A qiskit Macro.
    *~ Create countless adventure, legacy and tales. ~*
    """

    __name__ = "QurriumPrototype"
    """The name of Qurrium."""
    short_name = "qurrium"
    """The short name of Qurrium."""

    # Wave
    def add(
        self,
        wave: QuantumCircuit,
        key: Optional[Hashable] = None,
        replace: Literal[True, False, "duplicate"] = True,
    ) -> Hashable:
        """Add new wave function to measure.

        Args:
            wave (QuantumCircuit): The wave functions or circuits want to measure.
            key (Optional[Hashable], optional):
                Given a specific key to add to the wave function or circuit,
                if `key == None`, then generate a number as key.
                Defaults to `None`.
            replace (Literal[True, False, &#39;duplicate&#39;], optional):
                If the key is already in the wave function or circuit,
                then replace the old wave function or circuit when `True`,
                or duplicate the wave function or circuit when `'duplicate'`.
                Defaults to `True`.

        Returns:
            Optional[Hashable]: Key of given wave function in `.waves`.
        """
        return self.waves.add(wave=wave, key=key, replace=replace)

    def remove(self, key: Hashable) -> None:
        """Remove wave function from `.waves`.

        Args:
            wave (Hashable): The key of wave in `.waves`.
        """
        self.waves.remove(key)

    def has(self, wavename: Hashable) -> bool:
        """Is there a wave with specific name.

        Args:
            wavename (Hashable): Name of wave which is used in `.waves`

        Returns:
            bool: Exist or not.
        """
        return self.waves.has(wavename)

    @property
    @abstractmethod
    def experiment_instance(self) -> Type[_E]:
        """The instance of experiment."""
        raise NotImplementedError("The experiment is not defined.")

    def __init__(
        self,
    ) -> None:
        self.waves: WaveContainer = WaveContainer()
        """The wave functions container."""

        self.orphan_exps: ExperimentContainer[_E] = ExperimentContainer()
        """The orphan experiments container."""

        self.multimanagers: MultiManagerContainer[_E] = MultiManagerContainer()
        """The last multimanager be called."""

        self.accessor: Optional[RemoteAccessor] = None
        """The accessor of extra backend.
        It will be None if no extra backend is loaded.
        """

        self.passmanagers: PassManagerContainer = PassManagerContainer()
        """The collection of pass managers."""

        self.exps: ExperimentContainerWrapper[_E] = ExperimentContainerWrapper(
            orphan_exps=self.orphan_exps,
            multimanagers=self.multimanagers,
        )
        """The wrapper of experiments container from orphan_exps and multimanagers."""

        if hasattr(self, "__post_init__"):
            # Call the __post_init__ method if it exists
            getattr(self, "__post_init__")()

    def build(
        self,
        circuits: list[Union[QuantumCircuit, Hashable]],
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
        **custom_and_main_kwargs: Any,
    ) -> str:
        """Build the experiment.

        Args:
            circuits (list[Union[QuantumCircuit, Hashable]]):
                The circuits or keys of circuits in `.waves`.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The quantum backend. Defaults to AerSimulator().
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                The extra arguments for running the job.
                For :meth:`backend.run()` from :cls:`qiskit.providers.backend`. Defaults to `{}`.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to `None`.
            passmanager (Optional[Union[str, PassManager, tuple[str, PassManager]]], optional):
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Given the experiment multiple tags to make a dictionary for recongnizing it.

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
            custom_and_main_kwargs (Any):
                Other custom arguments.

        Returns:
            ExperimentPrototype: The experiment.
        """
        passmanager_pair = passmanager_processor(
            passmanager=passmanager, passmanager_container=self.passmanagers
        )
        targets = self.waves.process(circuits)

        new_exps = self.experiment_instance.build(
            targets=targets,
            shots=shots,
            backend=backend,
            exp_name=exp_name,
            run_args=run_args,
            transpile_args=transpile_args,
            passmanager_pair=passmanager_pair,
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
            **custom_and_main_kwargs,
        )

        self.orphan_exps[new_exps.commons.exp_id] = new_exps
        return new_exps.exp_id

    def output(
        self,
        # create new exp
        circuits: Optional[list[Union[QuantumCircuit, Hashable]]] = None,
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager: Optional[Union[str, PassManager, tuple[str, PassManager]]] = None,
        tags: Optional[tuple[str, ...]] = None,
        # already built exp
        exp_id: Optional[str] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
        **custom_and_main_kwargs: Any,
    ) -> str:
        """Output the experiment.

        Args:
            circuits (Optional[list[Union[QuantumCircuit, Hashable]]], optional):
                The circuits or keys of circuits in `.waves`.
                Defaults to None.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                Arguments for :meth:`Backend.run`. Defaults to `None`.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to `None`.
            passmanager (Optional[Union[str, PassManager, tuple[str, PassManager]], optional):
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Given the experiment multiple tags to make a dictionary for recongnizing it.
                Defaults to None.

            exp_id (Optional[str], optional):
                The ID of experiment. Defaults to None.

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
            custom_and_main_kwargs (Any):
                Other custom arguments.

        Returns:
            str: The experiment ID.
        """

        if exp_id is None and circuits is None:
            raise ValueError("No circuits or exp_id given.")
        if exp_id is not None and circuits is not None:
            raise ValueError("Both circuits and exp_id are given.")

        if exp_id is None:
            assert (
                circuits is not None
            ), "No circuits given, but it should be given for passing check."
            exp_id = self.build(
                circuits=circuits,
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
                **custom_and_main_kwargs,
            )

        runned_exp_id = self.orphan_exps[exp_id].run(pbar=pbar)

        resulted_exp_id = self.orphan_exps[runned_exp_id].result(
            export=export,
            save_location=save_location,
            mode=mode,
            indent=indent,
            encoding=encoding,
            jsonable=jsonable,
        )

        return resulted_exp_id

    @abstractmethod
    def measure_to_output(self) -> OutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form."""
        raise NotImplementedError("The method is not defined.")

    @abstractmethod
    def measure(self) -> str:
        """Execute the experiment."""
        raise NotImplementedError("The method is not defined.")

    # pylint: disable=invalid-name
    def multiBuild(
        self,
        config_list: list[Union[dict[str, Any], BasicArgs, Any]],
        summoner_name: str = short_name,
        summoner_id: Optional[str] = None,
        shots: int = 1024,
        backend: Backend = GeneralSimulator(),
        tags: Optional[tuple[str, ...]] = None,
        manager_run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        save_location: Union[Path, str] = Path("./"),
        jobstype: Union[Literal["local"], PendingTargetProviderLiteral] = "local",
        pending_strategy: PendingStrategyLiteral = "tags",
        skip_build_write: bool = False,
        multiprocess_build: bool = False,
        multiprocess_write: bool = False,
    ) -> str:
        """Build the multimanager.

        Args:
            config_list (list[dict[str, Any]]):
                The list of default configurations of multiple experiment.
            summoner_name (str, optional):
                Name for multimanager. Defaults to their coresponding :attr:`short_name`.
            summoner_id (Optional[str], optional):
                Id for multimanager. Defaults to None.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The backend to run. Defaults to GeneralSimulator().
            tags (Optional[tuple[str, ...]], optional):
                Tags of experiment of :cls:`MultiManager`. Defaults to None.
            manager_run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                The extra arguments for running the job,
                but for all experiments in the multimanager.
                For :meth:`backend.run()` from :cls:`qiskit.providers.backend`. Defaults to `{}`.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            jobstype (Union[Literal['local'], PendingTargetProviderLiteral], optional):
                Type of jobs to run multiple experiments.
                - jobstype: "local", "IBMQ", "IBM", "AWS_Bracket", "Azure_Q"
                Defaults to "local".
            pending_strategy (PendingStrategyLiteral, optional):
                Type of pending strategy.
                - pendingStrategy: "default", "onetime", "each", "tags"
                Defaults to "tags".
            skip_build_write (bool, optional):
                Whether to skip the file writing during the building.
                Defaults to False.
            multiprocess_build (bool, optional):
                Whether use multiprocess for building. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.
        Returns:
            str: The summoner_id of multimanager.
        """

        if summoner_id in self.multimanagers:
            assert isinstance(summoner_id, str), "summoner_id should be string here."
            return summoner_id
        if summoner_id is not None:
            raise ValueError("Unknow summoner_id in multimanagers.")

        output_allow_config_list = [self.measure_to_output(**config) for config in config_list]
        ready_config_list = []
        for config in output_allow_config_list:
            config_targets = {
                "targets": self.waves.process(config.pop("circuits")),  # type: ignore
                "passmanager_pair": passmanager_processor(
                    passmanager=config.pop("passmanager"), passmanager_container=self.passmanagers
                ),
            }
            config_targets.update(config)
            ready_config_list.append(config_targets)

        print("| MultiManager building...")
        current_multimanager = MultiManager.build(
            config_list=ready_config_list,
            experiment_instance=self.experiment_instance,
            summoner_name=summoner_name,
            shots=shots,
            backend=backend,
            tags=tags,
            manager_run_args=manager_run_args,
            jobstype=jobstype,
            pending_strategy=pending_strategy,
            save_location=save_location,
            skip_writing=skip_build_write,
            multiprocess_build=multiprocess_build,
            multiprocess_write=multiprocess_write,
        )
        assert len(current_multimanager.beforewards.pending_pool) == 0
        assert len(current_multimanager.beforewards.circuits_map) == 0
        assert len(current_multimanager.beforewards.job_id) == 0

        self.multimanagers[current_multimanager.summoner_id] = current_multimanager

        return current_multimanager.multicommons.summoner_id

    def multiOutput(
        self,
        config_list: list[Union[dict[str, Any], BasicArgs, Any]],
        summoner_name: str = short_name,
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
            config_list (list[dict[str, Any]]):
                The list of default configurations of multiple experiment.
            summoner_name (str, optional):
                Name for multimanager. Defaults to their coresponding :attr:`short_name`.
            summoner_id (Optional[str], optional):
                Id for multimanager. Defaults to None.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The backend to run. Defaults to GeneralSimulator().
            tags (Optional[tuple[str, ...]], optional):
                Tags of experiment of :cls:`MultiManager`. Defaults to None.
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

        if tags is None:
            tags = ()

        besummonned = self.multiBuild(
            config_list=config_list,
            shots=shots,
            backend=backend,
            tags=tags,
            manager_run_args=manager_run_args,
            summoner_name=summoner_name,
            summoner_id=summoner_id,
            save_location=save_location,
            jobstype="local",
            pending_strategy="tags",
            skip_build_write=skip_build_write,
            multiprocess_build=multiprocess_build,
            multiprocess_write=multiprocess_write,
        )
        current_multimanager = self.multimanagers[besummonned]
        assert current_multimanager.summoner_id == besummonned

        print("| MultiOutput running...")
        circ_serial: list[int] = []
        experiment_progress = qurry_progressbar(current_multimanager.exps.items())

        for exp_id, exp_instance in experiment_progress:
            experiment_progress.set_description_str("Experiments running...")

            current_id = exp_instance.run(pbar=experiment_progress)
            assert current_id == exp_id, f"exps_id output: {current_id} != exp_id: {exp_id}"

            current_id = exp_instance.result(
                export=False,
                save_location=current_multimanager.multicommons.save_location,
            )
            assert current_id == exp_id, f"exps_id output: {current_id} != exp_id: {exp_id}"

            circ_serial_len = len(circ_serial)
            tmp_circ_serial = [
                idx + circ_serial_len for idx, _ in enumerate(exp_instance.beforewards.circuit)
            ]
            circ_serial += tmp_circ_serial

            current_multimanager.beforewards.pending_pool[exp_id] = tmp_circ_serial
            current_multimanager.beforewards.circuits_map[exp_id] = tmp_circ_serial
            current_multimanager.beforewards.job_id.append((exp_id, "local"))

        current_multimanager.multicommons.datetimes.add_serial("output")

        if not skip_output_write:
            bewritten = self.multiWrite(besummonned, multiprocess_write=multiprocess_write)
            assert bewritten == besummonned

        return current_multimanager.multicommons.summoner_id

    def multiPending(
        self,
        config_list: list[dict[str, Any]],
        summoner_name: str = short_name,
        summoner_id: Optional[str] = None,
        shots: int = 1024,
        backend: Backend = GeneralSimulator(),
        provider: Optional[Any] = None,
        tags: Optional[tuple[str, ...]] = None,
        manager_run_args: Optional[dict[str, Any]] = None,
        save_location: Union[Path, str] = Path("./"),
        jobstype: PendingTargetProviderLiteral = "IBM",
        pending_strategy: PendingStrategyLiteral = "tags",
    ) -> str:
        """Pending the multiple experiments.

        Args:
            config_list (list[dict[str, Any]]):
                The list of default configurations of multiple experiment.
            summoner_name (str, optional):
                Name for multimanager. Defaults to their coresponding :attr:`short_name`.
            summoner_id (Optional[str], optional):
                Id for multimanager. Defaults to None.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The quantum backend. Defaults to GeneralSimulator().
            provider (Optional[Any], optional):
                The provider. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Tags of experiment of :cls:`MultiManager`. Defaults to None.
            manager_run_args (Optional[dict[str, Any]], optional):
                The extra arguments for running the job,
                but for all experiments in the multimanager.
                For :meth:`backend.run()` from :cls:`qiskit.providers.backend`. Defaults to `{}`.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            jobstype (PendingTargetProviderLiteral, optional):
                Type of jobs to run multiple experiments.
                - jobstype: "local", "IBMQ", "IBM", "AWS_Bracket", "Azure_Q"
                Defaults to "IBM".
            pending_strategy (PendingStrategyLiteral, optional):
                Type of pending strategy.
                - pendingStrategy: "default", "onetime", "each", "tags"
                Defaults to "tags".

        Returns:
            str: The summoner_id of multimanager.
        """

        besummonned = self.multiBuild(
            config_list=config_list,
            shots=shots,
            backend=backend,
            tags=tags,
            manager_run_args=manager_run_args,
            summoner_name=summoner_name,
            summoner_id=summoner_id,
            save_location=save_location,
            jobstype="local",
            pending_strategy="tags",
            skip_build_write=True,
        )
        current_multimanager = self.multimanagers[besummonned]
        assert current_multimanager.summoner_id == besummonned

        print("| MultiPending running...")

        self.accessor = RemoteAccessor(
            multimanager=current_multimanager,
            experiment_container=current_multimanager.exps,
            backend=backend,
            backend_type=jobstype,
            provider=provider,
        )
        self.accessor.pending(
            pending_strategy=pending_strategy,
        )
        bewritten = self.multiWrite(besummonned)
        assert bewritten == besummonned

        return current_multimanager.multicommons.summoner_id

    def multiAnalysis(
        self,
        summoner_id: str,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: Optional[
            dict[Hashable, Union[dict[str, Any], AnalyzeArgs, bool, Any]]
        ] = None,
        skip_write: bool = False,
        multiprocess_write: bool = False,
        **analysis_args: Any,
    ) -> str:
        """Run the analysis for multiple experiments.

        Args:
            summoner_id (str): The summoner_id of multimanager.
            analysis_name (str, optional):
                The name of analysis. Defaults to 'report'.
            no_serialize (bool, optional):
                Whether to serialize the analysis. Defaults to False.
            specific_analysis_args
                Optional[dict[Hashable, Union[dict[str, Any], bool]]], optional
            ):
                The specific arguments for analysis. Defaults to None.
            skip_write (bool, optional):
                Whether to skip the file writing during the analysis. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.
            analysis_args (Any):
                Other arguments for analysis.

        Returns:
            str: The summoner_id of multimanager.
        """
        if specific_analysis_args is None:
            specific_analysis_args = {}

        if summoner_id in self.multimanagers:
            current_multimanager = self.multimanagers[summoner_id]
        else:
            raise ValueError("No such summoner_id in multimanagers.")

        report_name = current_multimanager.analyze(
            analysis_name=analysis_name,
            no_serialize=no_serialize,
            specific_analysis_args=specific_analysis_args,
            **analysis_args,
        )
        print(f'| "{report_name}" has been completed.')

        if not skip_write:
            self.multiWrite(summoner_id=summoner_id, multiprocess_write=multiprocess_write)

        return current_multimanager.multicommons.summoner_id

    def multiWrite(
        self,
        summoner_id: str,
        save_location: Optional[Union[Path, str]] = None,
        compress: bool = False,
        compress_overwrite: bool = False,
        remain_only_compressed: bool = False,
        export_transpiled_circuit: bool = False,
        skip_before_and_after: bool = False,
        skip_exps: bool = False,
        skip_quantities: bool = False,
        multiprocess_write: bool = False,
    ) -> str:
        """Write the multimanager to the file.

        Args:
            summoner_id (str): The summoner_id of multimanager.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            compress (bool, optional):
                Whether to compress the export file.
                Defaults to False.
            compress_overwrite (bool, optional):
                Whether to overwrite the compressed file.
                Defaults to False.
            remain_only_compressed (bool, optional):
                Whether to remain only compressed file.
                Defaults to False.
            export_transpiled_circuit (bool, optional):
                Whether to export the transpiled circuit.
                Defaults to False.
            skip_before_and_after (bool, optional):
                Skip the beforewards and afterwards. Defaults to False.
            skip_exps (bool, optional):
                Skip the experiments. Defaults to False.
            skip_quantities (bool, optional):
                Skip the quantities container. Defaults to False.
            multiprocess_write (bool, optional):
                Whether to use multiprocess to write the file.

        Raises:
            ValueError: summoner_id not in multimanagers.

        Returns:
            str: The summoner_id of multimanager.
        """

        if summoner_id not in self.multimanagers:
            raise ValueError("No such summoner_id in multimanagers.", summoner_id)

        current_multimanager = self.multimanagers[summoner_id]
        assert current_multimanager.summoner_id == summoner_id

        current_multimanager.write(
            save_location=save_location,
            export_transpiled_circuit=export_transpiled_circuit,
            skip_before_and_after=skip_before_and_after,
            skip_exps=skip_exps,
            skip_quantities=skip_quantities,
            multiprocess=multiprocess_write,
        )

        if compress:
            current_multimanager.compress(
                compress_overwrite=compress_overwrite,
                remain_only_compressed=remain_only_compressed,
            )
        else:
            if compress_overwrite or remain_only_compressed:
                warnings.warn(
                    "'compressOverwrite' or 'remainOnlyCompressed' is set to True, "
                    + "but 'compress' is False."
                )

        return current_multimanager.multicommons.summoner_id

    def multiRead(
        self,
        summoner_name: str,
        save_location: Union[Path, str] = Path("./"),
        reload: bool = False,
        read_from_tarfile: bool = False,
    ) -> str:
        """Read the multimanager from the file.

        Args:
            summoner_name (str): Name for multimanager.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            reload (bool, optional):
                Whether to reload the multimanager.
                Defaults to False.
            read_from_tarfile (bool, optional):
                Whether to read from the tarfile.
                Defaults to False.

        Returns:
            str: The summoner_id of multimanager.
        """

        filtered_summoner_id_names = [
            (k, v.summoner_name) for k, v in self.multimanagers.items() if v == summoner_name
        ]
        if len(filtered_summoner_id_names) == 1:
            assert filtered_summoner_id_names[0][0] in self.multimanagers
            if not reload:
                print(
                    "| Found existed multimanager: "
                    + f"{filtered_summoner_id_names[0][0]}: {filtered_summoner_id_names[0][1]}."
                )
                return filtered_summoner_id_names[0][0]
            print("| MultiRead reloading...")
            del self.multimanagers[filtered_summoner_id_names[0][0]]
        elif len(filtered_summoner_id_names) > 1:
            raise ValueError(
                f"Multiple multimanager found for '{summoner_name}': {filtered_summoner_id_names}."
            )

        current_multimanager = MultiManager.read(
            summoner_name=summoner_name,
            experiment_instance=self.experiment_instance,
            save_location=save_location,
            is_read_or_retrieve=True,
            read_from_tarfile=read_from_tarfile,
        )
        self.multimanagers[current_multimanager.summoner_id] = current_multimanager

        return current_multimanager.multicommons.summoner_id

    def multiRetrieve(
        self,
        summoner_name: Optional[str] = None,
        summoner_id: Optional[str] = None,
        backend: Optional[Backend] = None,
        provider: Optional[Any] = None,
        save_location: Union[Path, str] = Path("./"),
        refresh: bool = False,
        overwrite: bool = False,
        reload: bool = False,
        read_from_tarfile: bool = False,
    ) -> str:
        """Retrieve the multiple experiments.

        Args:
            summoner_name (Optional[str], optional):
                Name for multimanager. Defaults to None.
            summoner_id (Optional[str], optional):
                Id for multimanager. Defaults to None.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            provider (Optional[Any], optional):
                The provider. Defaults to None.
            save_location (Union[Path, str], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then cancelled the file to be exported.
                Defaults to Path('./').
            refresh (bool, optional):
                Whether to refresh the retrieve. Defaults to False.
            overwrite (bool, optional):
                Whether to overwrite the retrieve. Defaults to False.
            reload (bool, optional):
                Whether to reload the multimanager. Defaults to False.
            read_from_tarfile (bool, optional):
                Whether to read from the tarfile. Defaults to False.

        Raises:
            ValueError: No summoner_name or summoner_id given.
            ValueError: Both summoner_name and summoner_id are given.
            ValueError: No such summoner_id in multimanagers.
            TypeError: summoner_name or summoner_id is not str.
            ValueError: No backend or provider given.

        Returns:
            str: The summoner_id of multimanager.
        """

        if summoner_name is None and summoner_id is None:
            raise ValueError("No summoner_name or summoner_id given.")
        if summoner_id is not None and summoner_name is not None:
            raise ValueError("Both summoner_name and summoner_id are given.")

        if isinstance(summoner_id, str):
            if summoner_id in self.multimanagers:
                current_multimanager = self.multimanagers[summoner_id]
                besummonned = summoner_id
            else:
                raise ValueError("No such summoner_id in multimanagers.", summoner_id)
        elif isinstance(summoner_name, str):
            besummonned = self.multiRead(
                summoner_name=summoner_name,
                save_location=save_location,
                reload=reload,
                read_from_tarfile=read_from_tarfile,
            )
            current_multimanager = self.multimanagers[besummonned]
            assert current_multimanager.summoner_id == besummonned
        else:
            raise TypeError(
                f"summoner_name: {summoner_name} with type: {type(summoner_name)} or "
                + f"summoner_id: {summoner_id} with type: {type(summoner_id)} is not str."
            )

        print("| MultiRetrieve running...")
        jobs_type = current_multimanager.multicommons.jobstype
        if backend is None and provider is None:
            raise ValueError("No backend or provider given.")

        self.accessor = RemoteAccessor(
            multimanager=current_multimanager,
            experiment_container=current_multimanager.exps,
            backend=backend,
            provider=provider,
            backend_type=jobs_type,
        )

        if jobs_type == "IBMQ":
            self.accessor.retrieve(
                overwrite=overwrite,
                refresh=refresh,
            )
        elif jobs_type in ["IBM", "IBMRuntime"]:
            self.accessor.retrieve(overwrite=overwrite)

        else:
            warnings.warn(
                f"Jobstype of '{besummonned}' is "
                + f"{current_multimanager.multicommons.jobstype} which is not supported."
            )
            return besummonned

        retrieve_times = retrieve_counter(current_multimanager.multicommons.datetimes)

        if retrieve_times > 0:
            if overwrite:
                print(f"| Retrieve {current_multimanager.summoner_name} overwrite.")
            else:
                print(f"| Retrieve skip for {current_multimanager.summoner_name} existed.")
                return besummonned
        else:
            print(f"| Retrieve {current_multimanager.summoner_name} completed.")
        bewritten = self.multiWrite(besummonned)
        assert bewritten == besummonned

        return current_multimanager.multicommons.summoner_id

    def __repr__(self):
        return (
            f"<{self.__name__}("
            f"waves={self.waves._repr_oneline()}, "
            f"exps={self.exps._repr_oneline()}, "
            f"multimanagers_num={len(self.multimanagers)}, "
            f"accessor={self.accessor}, "
            f"passmanagers_len={len(self.passmanagers)})>"
        )

    def _repr_pretty_(self, p, cycle):
        # pylint: disable=protected-access
        len_multimanagers = len(self.multimanagers)
        len_passmanagers = len(self.passmanagers)
        if cycle:
            p.text(f"<{self.__name__}(...)>")
        else:
            with p.group(2, f"<{self.__name__}(", ")>"):
                p.breakable()
                p.text(f"waves={self.waves._repr_oneline()},")
                p.breakable()
                p.text(f"exps={self.exps._repr_oneline()},")
                p.breakable()
                with p.group(
                    2, "multimanagers=MultiManagers({", "}" + f", num={len_multimanagers}), "
                ):
                    for i, (k, v) in enumerate(self.multimanagers.items()):
                        p.breakable()
                        p.text(f"'{k}': {v._repr_oneline_no_id()}")
                        if i != len_multimanagers - 1:
                            p.text(",")
                p.breakable()
                p.text(f"accessor={self.accessor},")
                p.breakable()
                with p.group(2, "passmanagers=PassManagers({", "}" + f", num={len_passmanagers})"):
                    for i, (k, v) in enumerate(self.passmanagers.items()):
                        p.breakable()
                        p.text(f"'{k}': {v}")
                        if i != len_passmanagers - 1:
                            p.text(",")
            p.break_()
        # pylint: enable=protected-access


# pylint: enable=invalid-name
