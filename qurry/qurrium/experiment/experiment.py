"""ExperimentPrototype - The instance of experiment (:mod:`qurry.qurrium.experiment.experiment`)"""

import os
import json
import warnings
from abc import abstractmethod, ABC
from typing import Union, Optional, Any, Type, Literal, Generic
from collections.abc import Hashable
from multiprocessing import get_context
from pathlib import Path
import tqdm

from qiskit import transpile, QuantumCircuit
from qiskit.providers import Backend, JobV1 as Job
from qiskit.transpiler.passmanager import PassManager

from .arguments import Commonparams, _A, create_exp_args, create_exp_commons, create_exp_outfields
from .beforewards import Before, create_beforewards
from .afterwards import After, create_afterwards
from .analyses import AnalysesContainer, _R
from .export import Export
from .utils import (
    exp_id_process,
    memory_usage_factor_expect,
    implementation_check,
    summonner_check,
    make_statesheet,
    create_save_location,
    decide_folder_and_filename,
)
from ..utils import get_counts_and_exceptions, qasm_dumps, outfields_check, outfields_hint
from ..utils.chunk import very_easy_chunk_size
from ...tools import (
    ParallelManager,
    DatetimeDict,
    set_pbar_description,
    backend_name_getter,
    DEFAULT_POOL_SIZE,
    qurry_progressbar,
    GeneralSimulator,
)
from ...capsule import quickJSON, DEFAULT_MODE, DEFAULT_ENCODING
from ...capsule.hoshi import Hoshi
from ...declare import RunArgsType, TranspileArgs
from ...exceptions import QurryResetSecurityActivated, QurryTranspileConfigurationIgnored


class ExperimentPrototype(ABC, Generic[_A, _R]):
    """The instance of experiment."""

    __name__ = "ExperimentPrototype"
    """Name of the QurryExperiment which could be overwritten."""

    @property
    @abstractmethod
    def arguments_instance(self) -> Type[_A]:
        """The arguments instance for this experiment."""
        raise NotImplementedError("This method should be implemented.")

    @property
    @abstractmethod
    def analysis_instance(self) -> Type[_R]:
        """The analysis instance for this experiment."""
        raise NotImplementedError("This method should be implemented.")

    @property
    def is_auto_analysis(self) -> bool:
        """Check if the experiment has auto analysis.

        Returns:
            bool: True if the experiment has auto analysis, False otherwise.
        """
        return len(self.analysis_instance.input_type()._fields) == 0

    @property
    def is_hold_by_multimanager(self) -> bool:
        """Check if the experiment is hold by a multimanager.

        Returns:
            bool: True if the experiment is hold by a multimanager, False otherwise.
        """
        return summonner_check(
            self.commons.serial, self.commons.summoner_id, self.commons.summoner_name
        )

    args: _A
    """The arguments of the experiment."""
    commons: Commonparams
    """The common parameters of the experiment."""
    outfields: dict[str, Any]
    """The outfields of the experiment."""
    beforewards: Before
    """The beforewards of the experiment."""
    afterwards: After
    """The afterwards of the experiment."""
    memory_usage_factor: int = -1
    """The factor of the memory usage of the experiment.
    When the experiment is created, it will be set to -1 for no measurement yet.
    When the experiment is built, it will be set to the memory usage of the experiment.

    The memory usage is estimated by the number of instructions in the circuits and
    the number of shots. The factor is calculated by the formula:

    .. code-block:: txt
        factor = target_circuit_instructions_num + sqrt(shots) * target_circuit_instructions_num

    where `target_circuit_instructions_num` is the number of instructions in the target circuits,
    `transpiled_circuit_instructions_num` is the number of instructions in the circuits
    which has been transpiled and will be run on the backend,
    and `shots` is the number of shots.

    The factor is rounded to the nearest integer.
    The factor is used to estimate the memory usage of the experiment.
    """

    def __init__(
        self,
        arguments: Union[_A, dict[str, Any]],
        commonparams: Union[Commonparams, dict[str, Any]],
        outfields: dict[str, Any],
        beforewards: Optional[Before] = None,
        afterwards: Optional[After] = None,
        reports: Optional[AnalysesContainer] = None,
    ) -> None:
        """Initialize the experiment.

        Args:
            arguments (Optional[Union[NamedTuple, dict[str, Any]]]):
                The arguments of the experiment.
            commonparams (Optional[Union[Commonparams, dict[str, Any]]]):
                The common parameters of the experiment.
            outfields (Optional[dict[str, Any]]): The outfields of the experiment.
            beforewards (Optional[Before], optional):
                The beforewards of the experiment. Defaults to None.
            afterwards (Optional[After], optional):
                The afterwards of the experiment. Defaults to None.
            reports (Optional[AnalysesContainer], optional):
                The reports of the experiment. Defaults to None.
        """
        self.args, arguments_deprecated = create_exp_args(arguments, self.arguments_instance)
        self.commons, commonparams_deprecated = create_exp_commons(commonparams)
        self.outfields = create_exp_outfields(outfields)
        # Add deprecated arguments to outfields only if they are not empty
        if len(arguments_deprecated):
            self.outfields["arguments_deprecated"] = arguments_deprecated
        if len(commonparams_deprecated):
            self.outfields["commonparams_deprecated"] = commonparams_deprecated
        implementation_check(self.__name__, self.args, self.commons)
        summonner_check(self.commons.serial, self.commons.summoner_id, self.commons.summoner_name)

        self.beforewards = create_beforewards(beforewards)
        self.afterwards = create_afterwards(afterwards)
        self.reports: AnalysesContainer[_R] = (
            reports if isinstance(reports, AnalysesContainer) else AnalysesContainer()
        )
        """The reports of the experiment."""

    @classmethod
    @abstractmethod
    def params_control(
        cls, targets: list[tuple[Hashable, QuantumCircuit]], exp_name: str, **custom_kwargs: Any
    ) -> tuple[_A, Commonparams, dict[str, Any]]:
        """Control the experiment's parameters.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]): The circuits of the experiment.
            exp_name (str):
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
            custom_kwargs (Any): Other custom arguments.

        Raises:
            NotImplementedError: This method should be implemented.
        """

        raise NotImplementedError("This method should be implemented.")

    @classmethod
    def _params_control_core(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_id: Optional[str] = None,
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: RunArgsType = None,
        transpile_args: Optional[TranspileArgs] = None,
        # multimanager
        tags: Optional[tuple[str, ...]] = None,
        serial: Optional[int] = None,
        summoner_id: Optional[Hashable] = None,
        summoner_name: Optional[str] = None,
        # process tool
        mute_outfields_warning: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
        **custom_kwargs: Any,
    ):
        """Control the experiment's general parameters.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]): The circuits of the experiment.
            exp_id (Optional[str], optional):
                If input is `None`, then create an new experiment.
                If input is a existed experiment ID, then use it.
                Otherwise, use the experiment with given specific ID.
                Defaults to None.
            shots (int, optional): Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional): The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Given the experiment multiple tags to make a dictionary for recongnizing it.
                Defaults to None.
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
            mute_outfields_warning (bool, optional):
                Mute the warning when there are unused arguments detected and stored in outfields.
                Defaults to False.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.
            custom_kwargs (Any):
                Other custom arguments.

        Returns:
            ExperimentPrototype: The experiment.
        """
        if run_args is None:
            run_args = {}
        if transpile_args is None:
            transpile_args = {}
        if backend is None:
            backend = GeneralSimulator()
        if tags is None:
            tags = ()

        # Given parameters and default parameters
        set_pbar_description(pbar, "Prepaing parameters...")

        arguments, commonparams, outfields = cls.params_control(
            targets=targets,
            exp_id=exp_id_process(exp_id),
            shots=shots,
            backend=backend,
            run_args=run_args,
            transpile_args=transpile_args,
            exp_name=exp_name,
            tags=tags,
            save_location=Path("./"),
            serial=serial,
            summoner_id=summoner_id,
            summoner_name=summoner_name,
            datetimes=DatetimeDict(),
            **custom_kwargs,
        )

        outfield_maybe, outfields_unknown = outfields_check(
            outfields, arguments._fields + commonparams._fields
        )
        outfields_hint(outfield_maybe, outfields_unknown, mute_outfields_warning)

        set_pbar_description(pbar, "Create experiment instance... ")
        new_exps = cls(arguments, commonparams, outfields)

        assert isinstance(new_exps.commons.backend, Backend), "Require a valid backend."
        assert len(new_exps.beforewards.circuit) == 0, "New experiment should have no circuit."
        assert len(new_exps.beforewards.circuit_qasm) == 0, "New experiment should have no qasm."
        assert len(new_exps.afterwards.result) == 0, "New experiment should have no result."
        assert len(new_exps.afterwards.counts) == 0, "New experiment should have no counts."

        return new_exps

    @classmethod
    @abstractmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: _A,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.
        Where should be overwritten by each construction of new measurement.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]): The circuits of the experiment.
            arguments (_Arg): The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.
            multiprocess (bool, optional): Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the outfields.
        """
        raise NotImplementedError("This method should be implemented.")

    @classmethod
    def build(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        shots: int = 1024,
        backend: Optional[Backend] = None,
        exp_name: str = "experiment",
        run_args: RunArgsType = None,
        transpile_args: Optional[TranspileArgs] = None,
        passmanager_pair: Optional[tuple[str, PassManager]] = None,
        tags: Optional[tuple[str, ...]] = None,
        # multimanager
        serial: Optional[int] = None,
        summoner_id: Optional[Hashable] = None,
        summoner_name: Optional[str] = None,
        # process tool
        qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
        **custom_and_main_kwargs: Any,
    ):
        """Construct the experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]): The circuits of the experiment.
            shots (int, optional): Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional): The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
                Defaults to None.
            passmanager_pair (Optional[tuple[str, PassManager]], optional):
                The passmanager pair for transpile. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                Given the experiment multiple tags to make a dictionary for recongnizing it.
                Defaults to None.

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
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

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
                pool.starmap(qasm_dumps, [(q, qasm_version) for q in cirqs])
            )
            current_exp.beforewards.target_qasm.extend(
                zip(
                    [str(k) for k in targets_keys],
                    pool.starmap(qasm_dumps, [(q, qasm_version) for q in targets_values]),
                )
            )
        else:
            current_exp.beforewards.circuit_qasm.extend(
                [qasm_dumps(q, qasm_version) for q in cirqs]
            )
            current_exp.beforewards.target_qasm.extend(
                zip(
                    [str(k) for k in targets_keys],
                    [qasm_dumps(q, qasm_version) for q in targets_values],
                )
            )

        # transpile
        if passmanager_pair is not None:
            passmanager_name, passmanager = passmanager_pair
            set_pbar_description(
                pbar, f"Circuit transpiling by passmanager '{passmanager_name}'..."
            )
            transpiled_circs = passmanager.run(
                circuits=cirqs, num_processes=None if multiprocess else 1  # type: ignore
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
            transpiled_circs: list[QuantumCircuit] = transpile(
                cirqs,
                backend=current_exp.commons.backend,
                num_processes=None if multiprocess else 1,
                **current_exp.commons.transpile_args,
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
            current_exp.write(save_location=save_location)

        return current_exp

    @classmethod
    def build_for_multiprocess(cls, config: dict[str, Any]):
        """Build wrapper for multiprocess.

        Args:
            config (dict[str, Any]): The arguments of the experiment.

        Returns:
            ExperimentPrototype: The experiment.
        """

        config.pop("multiprocess", None)
        config.pop("pbar", None)
        config["multiprocess"] = False
        return cls.build(**config), config

    # local execution
    def run(self, pbar: Optional[tqdm.tqdm] = None) -> str:
        """Export the result after running the job.

        Args:
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.

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

        set_pbar_description(pbar, "Executing...")
        event_name, date = self.commons.datetimes.add_serial("run")
        execution: Job = self.commons.backend.run(  # type: ignore
            self.beforewards.circuit, shots=self.commons.shots, **self.commons.run_args
        )
        # commons
        set_pbar_description(pbar, f"Executing completed '{event_name}', denoted date: {date}...")
        # beforewards
        self.beforewards.job_id.append(execution.job_id())
        # afterwards
        result = execution.result()
        self.afterwards.result.append(result)

        return self.exp_id

    def result(
        self,
        export: bool = False,
        save_location: Optional[Union[Path, str]] = None,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> str:
        """Export the result of the experiment.

        Args:
            export (bool, optional): Whether to export the experiment. Defaults to False.
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to None.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.

        Returns:
            str: The ID of the experiment.
        """

        if len(self.afterwards.result) == 0:
            raise ValueError("The job has not been executed yet.")
        assert len(self.afterwards.result) == 1, "The job has been executed more than once."

        set_pbar_description(pbar, "Result loading...")
        num = len(self.beforewards.circuit)
        counts, exceptions = get_counts_and_exceptions(result=self.afterwards.result[-1], num=num)
        if len(exceptions) > 0:
            if "exceptions" not in self.outfields:
                self.outfields["exceptions"] = {}
            for result_id, exception_item in exceptions.items():
                self.outfields["exceptions"][result_id] = exception_item

        set_pbar_description(pbar, "Counts loading...")
        self.afterwards.counts.extend(counts)

        if self.is_auto_analysis:
            if self.is_hold_by_multimanager:
                set_pbar_description(
                    pbar,
                    "Auto running analysis will take over by "
                    f"{self.commons.summoner_id}: "
                    f"{self.commons.summoner_name} after all experiments are done.",
                )
            else:
                set_pbar_description(pbar, "Running analysis for no input required...")
                self.analyze()

        if export:
            # export may be slow, consider export at finish or something
            if isinstance(save_location, (Path, str)):
                set_pbar_description(pbar, "Setup data exporting...")
                self.write(save_location=save_location)

        return self.exp_id

    # remote execution
    def _remote_result_taking(
        self,
        counts_tmp_container: dict[int, dict[str, int]],
        summoner_id: str,
        idx_circs: list[int],
        retrieve_times_name: str,
    ) -> list[dict[str, int]]:
        """Take the result from remote execution.

        Args:
            counts_tmp_container (dict[int, dict[str, int]]): The counts temporary container.
            summoner_id (str): The summoner ID.
            idx_circs (list[int]): The index of circuits.
            retrieve_times_name (str): The retrieve times name.
            current (str): The current time.

        Returns:
            list[dict[str, int]]: The counts.
        """
        if summoner_id == self.commons.summoner_id:
            self.afterwards.counts.clear()
            self.afterwards.result.clear()
            for idx in idx_circs:
                self.afterwards.counts.append(counts_tmp_container[idx])
            self.commons.datetimes.add_only(retrieve_times_name)
        else:
            warnings.warn(
                f"Summoner ID {summoner_id} is not equal to"
                + f" current summoner ID {self.commons.summoner_id}. "
                + "The counts will not be updated.",
                category=QurryResetSecurityActivated,
            )
        return self.afterwards.counts

    def replace_backend(self, backend: Backend) -> None:
        """Replace the backend of the experiment.

        Args:
            backend (Backend): The new backend.

        Raises:
            ValueError: If the new backend is not a valid backend.
            ValueError: If the new backend is not a runnable backend.
        """
        if not isinstance(backend, Backend):
            raise ValueError(f"Require a valid backend, but new backend: {backend} does not.")
        if not hasattr(backend, "run"):
            raise ValueError(f"Require a runnable backend, but new backend: {backend} does not.")

        old_backend = self.commons.backend
        old_backend_name = backend_name_getter(old_backend)
        new_backend_name = backend_name_getter(backend)
        self.commons.datetimes.add_serial(f"replace-{old_backend_name}-to-{new_backend_name}")
        self.commons = self.commons._replace(backend=backend)

    def __getitem__(self, key) -> Any:
        if key in self.beforewards._fields:
            return getattr(self.beforewards, key)
        if key in self.afterwards._fields:
            return getattr(self.afterwards, key)
        raise KeyError(
            f"{key} is not a valid field of " + f"'{Before.__name__}' and '{After.__name__}'."
        )

    # analysis
    @classmethod
    @abstractmethod
    def quantities(cls) -> dict[str, Any]:
        """Computing specific squantity.
        Where should be overwritten by each construction of new measurement.
        """

    @abstractmethod
    def analyze(self) -> _R:
        """Analyzing the example circuit results in specific method.
        Where should be overwritten by each construction of new measurement.

        If the analysis requires additional parameters,
        they should be passed as arguments to this method.
        Also, they should be defined in the :meth:`input_type` in the :cls:`AnalysisPrototype`
        for :meth:`result` will count the input fields from the analysis to determine
        whether to call this method for no input required.

        Returns:
            _R: The result of the analysis.
        """
        raise NotImplementedError("This method should be implemented.")

    # show info
    def __hash__(self) -> int:
        return hash(self.commons.exp_id)

    @property
    def exp_id(self) -> str:
        """ID of experiment."""
        return self.commons.exp_id

    def __repr__(self) -> str:
        return (
            f"<{self.__name__}(exp_id={self.commons.exp_id}, {self.args}, {self.commons}, "
            f"unused_args_num={len(self.outfields)}, analysis_num={len(self.reports)})>"
        )

    def _repr_no_id(self) -> str:
        return (
            f"<{self.__name__}({self.args}, {self.commons}, "
            f"unused_args_num={len(self.outfields)}, analysis_num={len(self.reports)})>"
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(
                f"<{self.__name__}(exp_id={self.commons.exp_id}, {self.args}, {self.commons}, "
                f"unused_args_num={len(self.outfields)}, analysis_num={len(self.reports)})>"
            )
        else:
            with p.group(2, f"<{self.__name__}(", ")>"):
                p.text(f"exp_id={self.commons.exp_id}, ")
                p.breakable()
                p.text(f"{self.args},")
                p.breakable()
                p.text(f"{self.commons},")
                p.breakable()
                p.text(f"unused_args_num={len(self.outfields)},")
                p.breakable()
                p.text(f"analysis_num={len(self.reports)})")

    def statesheet(self, report_expanded: bool = False, hoshi: bool = False) -> Hoshi:
        """Show the state of experiment.

        Args:
            report_expanded (bool, optional): Show more infomation. Defaults to False.
            hoshi (bool, optional): Showing name of Hoshi. Defaults to False.

        Returns:
            Hoshi: Statesheet of experiment.
        """

        return make_statesheet(
            exp_name=self.__name__,
            args=self.args,
            commons=self.commons,
            outfields=self.outfields,
            beforewards=self.beforewards,
            afterwards=self.afterwards,
            reports=self.reports,
            report_expanded=report_expanded,
            hoshi=hoshi,
        )

    def export(
        self,
        save_location: Optional[Union[Path, str]] = None,
        export_transpiled_circuit: bool = False,
    ) -> Export:
        """Export the data of experiment into specific namedtuples for exporting.

        Args:
            save_location (Optional[Union[Path, str]], optional):
                The location to save the experiment. Defaults to None.
            export_transpiled_circuit (bool, optional):
                Whether to export the transpiled circuit as txt. Defaults to False.
                When set to True, the transpiled circuit will be exported as txt.
                Otherwise, the circuit will be not exported but circuit qasm remains.

        Returns:
            Export: A namedtuple containing the data of experiment
                which can be more easily to export as json file.
        """
        save_location = create_save_location(save_location, self.commons)
        if self.commons.save_location != save_location:
            self.commons = self.commons._replace(save_location=save_location)

        adventures, tales = self.beforewards.export(export_transpiled_circuit)
        legacy = self.afterwards.export()
        reports, tales_reports = self.reports.export()

        # multi-experiment mode
        folder, filename = decide_folder_and_filename(self.commons, self.args)
        files = {
            "folder": folder,
            "qurryinfo": folder + "qurryinfo.json",
            "args": folder + f"args/{filename}.args.json",
            "advent": folder + f"advent/{filename}.advent.json",
            "legacy": folder + f"legacy/{filename}.legacy.json",
        }
        for k in tales:
            files[f"tales.{k}"] = folder + f"tales/{filename}.{k}.json"
        files["reports"] = folder + f"reports/{filename}.reports.json"
        for k in tales_reports:
            files[f"reports.tales.{k}"] = folder + f"tales/{filename}.{k}.reports.json"

        return Export(
            exp_id=str(self.commons.exp_id),
            exp_name=str(self.args.exp_name),
            serial=(None if self.commons.serial is None else int(self.commons.serial)),
            summoner_id=(None if self.commons.summoner_id else str(self.commons.summoner_id)),
            summoner_name=(None if self.commons.summoner_name else str(self.commons.summoner_name)),
            filename=str(filename),
            files={k: str(Path(v)) for k, v in files.items()},
            args=self.args._asdict(),
            commons=self.commons.export(),
            outfields=self.outfields,
            adventures=adventures,
            legacy=legacy,
            tales=tales,
            reports=reports,
            tales_reports=tales_reports,
        )

    def write(
        self,
        save_location: Optional[Union[Path, str]] = None,
        export_transpiled_circuit: bool = False,
        qurryinfo_hold_access: Optional[str] = None,
        multiprocess: bool = True,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> tuple[str, dict[str, str]]:
        """Export the experiment data, if there is a previous export, then will overwrite.

        Args:
            save_location (Optional[Union[Path, str]], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then use the value in `self.commons` to be exported,
                if it's None too, then raise error. Defaults to None.
            export_transpiled_circuit (bool, optional):
                Whether to export the transpiled circuit as txt. Defaults to False.
                When set to True, the transpiled circuit will be exported as txt.
                Otherwise, the circuit will be not exported but circuit qasm remains.
            qurryinfo_hold_access (str, optional):
                Whether to hold the I/O of `qurryinfo`, then export by :cls:`MultiManager`,
                it should be control by :cls:`MultiManager`. Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.

        Returns:
            tuple[str, dict[str, str]]: The id of the experiment and the files location.
        """
        set_pbar_description(pbar, "Preparing to export...")

        # experiment write
        export_material = self.export(save_location, export_transpiled_circuit)
        exp_id, files = export_material.write(multiprocess, pbar)

        assert "qurryinfo" in files, "qurryinfo location is not in files."
        # qurryinfo write
        real_save_location = Path(self.commons.save_location)
        if (
            qurryinfo_hold_access == self.commons.summoner_id
            and self.commons.summoner_id is not None
        ):
            # if qurryinfo_hold_access is set, then export by MultiManager
            return exp_id, files
        qurryinfo_location = real_save_location / files["qurryinfo"]

        if os.path.exists(qurryinfo_location):
            with open(qurryinfo_location, "r", encoding=DEFAULT_ENCODING) as f:
                qurryinfo_found: dict[str, dict[str, str]] = dict(json.load(f))
                qurryinfo_found[exp_id] = files
            quickJSON(qurryinfo_found, str(qurryinfo_location), DEFAULT_MODE)
        else:
            quickJSON({exp_id: files}, str(qurryinfo_location), DEFAULT_MODE)

        return exp_id, files

    @classmethod
    def _read_core(
        cls,
        exp_id: str,
        file_index: dict[str, str],
        save_location: Union[Path, str] = Path("./"),
    ):
        """Core of read function.

        Args:
            exp_id (str): The id of the experiment to be read.
            file_index (dict[str, str]): The index of the experiment to be read.
            save_location (Union[Path, str]): The location of the experiment to be read.

        Raises:
            ValueError: 'save_location' needs to be the type of 'str' or 'Path'.
            FileNotFoundError: When `save_location` is not available.

        Returns:
            QurryExperiment: The experiment to be read.
        """

        save_location = create_save_location(save_location)
        if not os.path.exists(save_location):
            raise FileNotFoundError(f"'save_location' does not exist, '{save_location}'.")

        reading_return_args = Commonparams.read_with_arguments(
            exp_id=exp_id, file_index=file_index, save_location=save_location
        )
        exp_instance = cls(
            **reading_return_args,
            beforewards=Before.read(file_index=file_index, save_location=save_location),
            afterwards=After.read(file_index=file_index, save_location=save_location),
            reports=AnalysesContainer(),
        )
        reports_read = exp_instance.analysis_instance.read(
            file_index=file_index, save_location=save_location
        )
        exp_instance.reports.update(reports_read)

        return exp_instance

    @classmethod
    def _read_core_multiprocess(cls, all_arugments: tuple[str, dict[str, str], Union[Path, str]]):
        """Core of read function for multiprocess.

        Args:
            all_arugments (tuple[str, dict[str, str], Union[Path, str], str]):
                The arguments of the experiment to be read.
                - exp_id (str): The id of the experiment to be read.
                - file_index (dict[str, str]): The index of the experiment to be read.
                - save_location (Union[Path, str]): The location of the experiment to be read.

        Returns:
            QurryExperiment: The experiment to be read.
        """
        return cls._read_core(*all_arugments)

    @classmethod
    def read(
        cls,
        name_or_id: Union[Path, str],
        save_location: Union[Path, str] = Path("./"),
    ):
        """Read the experiment from file.

        Args:
            name_or_id (Union[Path, str]): The name or id of the experiment to be read.
            save_location (Union[Path, str], optional):
                The location of the experiment to be read. Defaults to Path('./').

        Raises:
            ValueError: 'save_location' needs to be the type of 'str' or 'Path'.
            FileNotFoundError: When `save_location` is not available.

        Returns:
            list[ExperimentPrototype]: The experiment to be read.
        """

        save_location = create_save_location(save_location)
        if not os.path.exists(save_location):
            raise FileNotFoundError(f"'save_location' does not exist, '{save_location}'.")
        export_location = save_location / name_or_id
        if not os.path.exists(export_location):
            raise FileNotFoundError(f"'ExportLoaction' does not exist, '{export_location}'.")
        qurryinfo_location = export_location / "qurryinfo.json"
        if not os.path.exists(qurryinfo_location):
            raise FileNotFoundError(
                f"'qurryinfo.json' does not exist at '{save_location}'. "
                + "It's required for loading all experiment data."
            )

        qurryinfo: dict[str, dict[str, str]] = {}
        with open(qurryinfo_location, "r", encoding=DEFAULT_ENCODING) as f:
            qurryinfo_found: dict[str, dict[str, str]] = json.load(f)
            qurryinfo.update(qurryinfo_found)

        num_exps = len(qurryinfo)
        chunks_num = very_easy_chunk_size(
            tasks_num=num_exps,
            num_process=DEFAULT_POOL_SIZE,
            max_chunk_size=min(max(1, num_exps // DEFAULT_POOL_SIZE), 40),
        )
        reading_pool = get_context("spawn").Pool(
            processes=DEFAULT_POOL_SIZE, maxtasksperchild=chunks_num * 2
        )
        with reading_pool as pool:
            exps_iterable = qurry_progressbar(
                pool.imap_unordered(
                    cls._read_core_multiprocess,
                    (
                        (exp_id, file_index, save_location)
                        for exp_id, file_index in qurryinfo.items()
                    ),
                ),
                total=num_exps,
                desc=f"Loading {num_exps} experiments ...",
            )
            exps = list(exps_iterable)

        return exps
