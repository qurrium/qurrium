"""StringOperator - Qurry
(:mod:`qurry.qurries.string_operator.qurry`)

"""

from pathlib import Path
from typing import Union, Optional, Any, Type, Literal
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler.passmanager import PassManager

from .utils import AvailableStringOperatorTypes
from .arguments import (
    SHORT_NAME,
    StringOperatorMeasureArgs,
    StringOperatorOutputArgs,
    StringOperatorAnalyzeArgs,
)
from .experiment import StringOperatorExperiment
from ...qurrium.qurrium import QurriumPrototype
from ...tools.backend import GeneralSimulator
from ...declare import BaseRunArgs, TranspileArgs

from ...tools.except_decorator import unproven_feature


@unproven_feature(message="The StringOperator is not proven, we can not guarantee the correctness.")
class StringOperator(QurriumPrototype[StringOperatorExperiment]):
    """String Operator Order

    Reference:
        .. note::
            - Crossing a topological phase transition with a quantum computer -
            Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank,
            [PhysRevResearch.4.L022020](https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020)

        .. code-block:: bibtex
            @article{PhysRevResearch.4.L022020,
                title = {Crossing a topological phase transition with a quantum computer},
                author = {Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank},
                journal = {Phys. Rev. Research},
                volume = {4},
                issue = {2},
                pages = {L022020},
                numpages = {8},
                year = {2022},
                month = {Apr},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevResearch.4.L022020},
                url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020}
            }

    """

    __name__ = "StringOperator"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> Type[StringOperatorExperiment]:
        """The container class responding to this Qurrium class."""
        return StringOperatorExperiment

    def measure_to_output(
        self,
        wave: Optional[Union[QuantumCircuit, Hashable]] = None,
        i: Optional[int] = None,
        k: Optional[int] = None,
        str_op: AvailableStringOperatorTypes = "i",
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
    ) -> StringOperatorOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            i (Optional[int], optional):
                The index of beginning qubits in the quantum circuit.
            k (Optional[int], optional):
                The index of ending qubits in the quantum circuit.
            str_op (AvailableStringOperatorTypes, optional):
                The string operator.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
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
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to None.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
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

        Returns:
            StringOperatorOutputArgs: The output arguments.
        """
        if wave is None:
            raise ValueError("The `wave` must be provided.")

        return {
            "circuits": [wave],
            "i": i,
            "k": k,
            "str_op": str_op,
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
        }

    def measure(
        self,
        wave: Optional[Union[QuantumCircuit, Hashable]] = None,
        i: Optional[int] = None,
        k: Optional[int] = None,
        str_op: AvailableStringOperatorTypes = "i",
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
            wave (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            i (Optional[int], optional):
                The index of beginning qubits in the quantum circuit.
            k (Optional[int], optional):
                The index of ending qubits in the quantum circuit.
            str_op (AvailableStringOperatorTypes, optional):
                The string operator.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
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
                The passmanager. Defaults to None.
            tags (Optional[tuple[str, ...]], optional):
                The tags of the experiment. Defaults to None.

            qasm_version (Literal["qasm2", "qasm3"], optional):
                The version of OpenQASM. Defaults to "qasm3".
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

        Returns:
            str: The ID of the experiment
        """

        output_args = self.measure_to_output(
            wave=wave,
            i=i,
            k=k,
            str_op=str_op,
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
        config_list: list[Union[dict[str, Any], StringOperatorMeasureArgs]],
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
            config_list (list[Union[dict[str, Any], StringOperatorMeasureArgs]]):
                The list of default configurations of multiple experiment. Defaults to [].
            summoner_name (str, optional):
                Name for multimanager. Defaults to 'exps'.
            summoner_id (Optional[str], optional):
                Name for multimanager. Defaults to None.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Backend, optional):
                The quantum backend.
                Defaults to AerSimulator().
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
            multiprocess_build=multiprocess_build,
            multiprocess_write=multiprocess_write,
        )

    def multiAnalysis(
        self,
        summoner_id: str,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: Optional[
            dict[Hashable, Union[dict[str, Any], StringOperatorAnalyzeArgs, bool]]
        ] = None,
        skip_write: bool = False,
        multiprocess_write: bool = False,
        # analysis arguments
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
                    dict[str, Any], StringOperatorAnalyzeArgs, bool
                ]]], optional
            ):
                The specific arguments for analysis. Defaults to None.
            skip_write (bool, optional):
                Whether to skip the file writing during the analysis. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

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
            **analysis_args,
        )
