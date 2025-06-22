"""StringOperator - Qurrium (:mod:`qurry.qurries.string_operator.qurry`)"""

from pathlib import Path
from typing import Union, Optional, Type, Literal
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .utils import StringOperatorLibType, StringOperatorDirection
from .arguments import (
    SHORT_NAME,
    StringOperatorMeasureArgs,
    StringOperatorOutputArgs,
    StringOperatorAnalyzeArgs,
)
from .experiment import StringOperatorExperiment
from ...qurrium import QurriumPrototype
from ...declare import RunArgsType, TranspileArgs, PassManagerType


class StringOperator(
    QurriumPrototype[
        StringOperatorExperiment,
        StringOperatorMeasureArgs,
        StringOperatorOutputArgs,
        StringOperatorAnalyzeArgs,
    ]
):
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
        str_op: StringOperatorLibType = "i",
        on_dir: StringOperatorDirection = "x",
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
    ) -> StringOperatorOutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            i (Optional[int], optional):
                The index of beginning qubits in the quantum circuit.
            k (Optional[int], optional):
                The index of ending qubits in the quantum circuit.
            str_op (StringOperatorLibType, optional):
                The string operator. Defaults to "i".
            on_dir (StringOperatorDirection, optional):
                The direction of the string operator, either 'x' or 'y'. Defaults to "x".
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
            StringOperatorOutputArgs: The output arguments.
        """
        if wave is None:
            raise ValueError("The `wave` must be provided.")

        return {
            "circuits": [wave],
            "i": i,
            "k": k,
            "str_op": str_op,
            "on_dir": on_dir,
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
        i: Optional[int] = None,
        k: Optional[int] = None,
        str_op: StringOperatorLibType = "i",
        on_dir: StringOperatorDirection = "x",
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

        Args:
            wave (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            i (Optional[int], optional):
                The index of beginning qubits in the quantum circuit.
            k (Optional[int], optional):
                The index of ending qubits in the quantum circuit.
            str_op (StringOperatorLibType, optional):
                The string operator. Defaults to "i".
            on_dir (StringOperatorDirection, optional):
                The direction of the string operator, either 'x' or 'y'. Defaults to "x".
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
            i=i,
            k=k,
            str_op=str_op,
            on_dir=on_dir,
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
