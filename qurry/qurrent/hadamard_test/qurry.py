"""EntropyMeasureHadamard - Qurrium
(:mod:`qurry.qurrent.hadamard_test.qurry`)

"""

from pathlib import Path
from typing import Union, Optional, Type, Literal
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .arguments import (
    SHORT_NAME,
    EntropyMeasureHadamardMeasureArgs,
    EntropyMeasureHadamardOutputArgs,
    EntropyMeasureHadamardAnalyzeArgs,
)
from .experiment import EntropyMeasureHadamardExperiment
from ...qurrium import QurriumPrototype
from ...declare import RunArgsType, TranspileArgs, PassManagerType


class EntropyMeasureHadamard(
    QurriumPrototype[
        EntropyMeasureHadamardExperiment,
        EntropyMeasureHadamardMeasureArgs,
        EntropyMeasureHadamardOutputArgs,
        EntropyMeasureHadamardAnalyzeArgs,
    ]
):
    """Hadamard test for entanglement entropy.

    - Which entropy:

        The entropy we compute is the Second Order Rényi Entropy.

    """

    __name__ = "EntropyMeasureHadamard"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> Type[EntropyMeasureHadamardExperiment]:
        """The container class responding to this Qurrium class."""
        return EntropyMeasureHadamardExperiment

    def measure_to_output(
        self,
        wave: Optional[Union[QuantumCircuit, Hashable]] = None,
        degree: Optional[Union[int, tuple[int, int]]] = None,
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
    ) -> EntropyMeasureHadamardOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave (Union[QuantumCircuit, Hashable]):
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
            EntropyMeasureHadamardOutputArgs: The output arguments.
        """
        if wave is None:
            raise ValueError("The `wave` must be provided.")

        return {
            "circuits": [wave],
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
        wave: Optional[Union[QuantumCircuit, Hashable]] = None,
        degree: Optional[Union[int, tuple[int, int]]] = None,
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
            wave (Union[QuantumCircuit, Hashable]):
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
            str: The ID of the experiment
        """

        output_args = self.measure_to_output(
            wave=wave,
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
