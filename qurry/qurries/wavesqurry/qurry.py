"""WavesExecuter (:mod:`qurry.qurries.wavesqurry`)

It is only for pendings and retrieve to remote backend.
"""

from typing import Union, Optional, Literal
from pathlib import Path
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .arguments import SHORT_NAME, WEMeasureArgs, WEOutputArgs
from .experiment import WEExperiment
from ..samplingqurry.analysis import DummyAnalyzeArgs
from ...qurrium import QurriumPrototype, RunArgsType, TranspileArgs, PassManagerType, WCKeyable


class WavesExecuter(QurriumPrototype[WEExperiment, WEMeasureArgs, WEOutputArgs, DummyAnalyzeArgs]):
    """The pending and retrieve executer for waves."""

    __name__ = "WavesExecuter"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> type[WEExperiment]:
        """The container class responding to this Qurrium class."""
        return WEExperiment

    def measure_to_output(
        self,
        waves: Optional[list[Union[QuantumCircuit, WCKeyable]]] = None,
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
    ) -> WEOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            waves (list[Union[QuantumCircuit, WCKeyable]]):
                The key or the circuit to execute.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`~qiskit.compiler.transpile`.
                Defaults to None.
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
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.

        Returns:
            WavesExecuterOutputArgs: The output arguments.
        """
        if waves is None:
            raise ValueError("The `waves` must be provided.")

        return {
            "circuits": waves,
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
        waves: Optional[list[Union[QuantumCircuit, WCKeyable]]] = None,
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
    ):
        """Execute the experiment.

        Args:
            waves (list[Union[QuantumCircuit, WCKeyable]]):
                The key or the circuit to execute.
            shots (int, optional):
                Shots of the job. Defaults to `1024`.
            backend (Optional[Backend], optional):
                The quantum backend. Defaults to None.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            run_args (RunArgsType, optional):
                Arguments for :meth:`Backend.run`. Defaults to None.
            transpile_args (Optional[TranspileArgs], optional):
                Arguments of :func:`~qiskit.compiler.transpile`.
                Defaults to None.
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
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.

        Returns:
            str: The experiment ID.
        """

        output_args = self.measure_to_output(
            waves=waves,
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
