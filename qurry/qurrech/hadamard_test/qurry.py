"""EchoListenHadamard - Qurrium
(:mod:`qurry.qurrech.hadamard_test.qurry`)

"""

from pathlib import Path
from typing import Union, Optional, Type, Literal
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .arguments import (
    SHORT_NAME,
    EchoListenHadamardMeasureArgs,
    EchoListenHadamardOutputArgs,
    EchoListenHadamardAnalyzeArgs,
)
from .experiment import EchoListenHadamardExperiment
from ...qurrium import QurriumPrototype
from ...declare import RunArgsType, TranspileArgs, PassManagerType


class EchoListenHadamard(
    QurriumPrototype[
        EchoListenHadamardExperiment,
        EchoListenHadamardMeasureArgs,
        EchoListenHadamardOutputArgs,
        EchoListenHadamardAnalyzeArgs,
    ]
):
    """The experiment for calculating entangled entropy with more information combined."""

    __name__ = "EchoHadamardTest"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> Type[EchoListenHadamardExperiment]:
        """The experiment instance for this experiment."""
        return EchoListenHadamardExperiment

    def measure_to_output(
        self,
        wave1: Optional[Union[QuantumCircuit, Hashable]] = None,
        wave2: Optional[Union[QuantumCircuit, Hashable]] = None,
        degree: Union[int, tuple[int, int], None] = None,
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
    ) -> EchoListenHadamardOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            degree (Optional[Union[int, tuple[int, int]]], optional):
                The degree of the experiment.
                Defaults to None.
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
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
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
            EchoListenHadamardOutputArgs: The output arguments.
        """
        if wave1 is None:
            raise ValueError("The `wave` must be provided.")
        if wave2 is None:
            raise ValueError("The `wave2` must be provided.")

        return {
            "circuits": [wave1, wave2],
            "degree": degree,
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
        wave1: Optional[Union[QuantumCircuit, Hashable]] = None,
        wave2: Optional[Union[QuantumCircuit, Hashable]] = None,
        degree: Union[int, tuple[int, int], None] = None,
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

            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            degree (Optional[Union[int, tuple[int, int]]], optional):
                The degree of the experiment.
                Defaults to None.
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
                Arguments of :func:`transpile` from :mod:`qiskit.compiler.transpiler`.
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
            str: The ID of the experiment.
        """

        output_args = self.measure_to_output(
            wave1=wave1,
            wave2=wave2,
            degree=degree,
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
