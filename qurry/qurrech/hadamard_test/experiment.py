"""EchoListenHadamard - Experiment (:mod:`qurry.qurrech.hadamard_test.experiment`)"""

from typing import Optional, Any
import tqdm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .analysis import ELHAnalysis
from .arguments import ELHArguments, SHORT_NAME
from ...qurrium import ExperimentPrototype, Commonparams, WCKeyable
from ...process.utils import qubit_selector


class ELHxperiment(ExperimentPrototype[ELHArguments, ELHAnalysis]):
    """The instance of experiment."""

    __name__ = "ELHxperiment"

    @classmethod
    def arguments_type(cls) -> type[ELHArguments]:
        """The arguments instance for this experiment."""
        return ELHArguments

    @classmethod
    def analysis_type(cls) -> type[ELHAnalysis]:
        """The analysis instance for this experiment."""
        return ELHAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[WCKeyable, QuantumCircuit]],
        exp_name: str = "exps",
        degree: Optional[tuple[int, int]] = None,
        **custom_kwargs: Any,
    ) -> tuple[ELHArguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[WCKeyable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'experiment'`.
            degree (Optional[tuple[int, int]], optional):
                The degree range.
                Defaults to None.
            custom_kwargs (Any):
                The custom parameters.

        Raises:
            ValueError: The number of target circuits should be 2.
            ValueError: If the number of qubits in two circuits is not the same.

        Returns:
            tuple[EntropyMeasureHadamardArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) != 2:
            raise ValueError("The number of target circuits should be 2.")

        target_key_01, target_circuit_01 = targets[0]
        num_qubits_01 = target_circuit_01.num_qubits
        target_key_02, target_circuit_02 = targets[1]
        num_qubits_02 = target_circuit_02.num_qubits

        if num_qubits_01 != num_qubits_02:
            raise ValueError(
                "The number of qubits in two circuits should be the same, "
                + f"but got {target_key_01}: {num_qubits_01} and {target_key_02}: {num_qubits_02}."
            )

        degree = qubit_selector(num_qubits_01, degree=degree)

        exp_name = f"{exp_name}.degree_{degree[0]}_{degree[1]}.{SHORT_NAME}"

        return ELHArguments.filter(
            exp_name=exp_name,
            target_keys=[target_key_01, target_key_02],
            degree=degree,
            **custom_kwargs,
        )

    @classmethod
    def method(
        cls,
        targets: list[tuple[WCKeyable, QuantumCircuit]],
        arguments: ELHArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[WCKeyable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (EchoListenHadamardArguments):
                The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment.
                Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the arguments of the experiment.
        """

        assert isinstance(arguments.degree, tuple), (
            f"The degree should be a tuple, got {arguments.degree}."
        )

        target_key_01, target_circuit_01 = targets[0]
        target_key_01 = "" if isinstance(target_key_01, int) else str(target_key_01)
        num_qubits_01 = target_circuit_01.num_qubits
        old_name_01 = "" if isinstance(target_circuit_01.name, str) else target_circuit_01.name
        target_key_02, target_circuit_02 = targets[1]
        target_key_02 = "" if isinstance(target_key_02, int) else str(target_key_02)
        num_qubits_02 = target_circuit_02.num_qubits
        old_name_02 = "" if isinstance(target_circuit_02.name, str) else target_circuit_02.name

        assert num_qubits_01 == num_qubits_02, (
            "The number of qubits in two circuits should be the same."
        )

        q_ancilla = QuantumRegister(1, "ancilla_1")
        q_func1 = QuantumRegister(num_qubits_01, "q1")
        q_func2 = QuantumRegister(num_qubits_01, "q2")
        c_meas1 = ClassicalRegister(1, "c1")
        qc_exp1 = QuantumCircuit(q_ancilla, q_func1, q_func2, c_meas1)
        qc_exp1.name = (
            f"{arguments.exp_name}" + ""
            if len(target_key_01) < 1 and len(target_key_02) < 1
            else (
                f".{target_key_01}_{target_key_02}" + ""
                if len(old_name_01) < 1 and len(old_name_02) < 1
                else f".{old_name_01}_{old_name_02}"
            )
        )

        qc_exp1.compose(
            target_circuit_01,
            [q_func1[i] for i in range(num_qubits_01)],
            inplace=True,
        )

        qc_exp1.compose(
            target_circuit_02,
            [q_func2[i] for i in range(num_qubits_01)],
            inplace=True,
        )

        qc_exp1.barrier()
        qc_exp1.h(q_ancilla)
        for i in range(*arguments.degree):
            qc_exp1.cswap(q_ancilla[0], q_func1[i], q_func2[i])
        qc_exp1.h(q_ancilla)
        qc_exp1.measure(q_ancilla, c_meas1)

        return [qc_exp1], {}

    def analyze(self, pbar: Optional[tqdm.tqdm] = None) -> ELHAnalysis:
        """Calculate the analysis of wave function overlap.

        Args:
            degree (Union[tuple[int, int], int]): Degree of the subsystem.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            dict[str, float]: A dictionary contains
                purity, entropy.
        """

        if pbar is not None:
            pbar.set_description("Calculating wave function overlap")

        analysis = self.analysis_type().perform_analysis(
            arguments=self.args,
            commonparams=self.commons,
            counts=self.afterwards.counts,
            analyze_arguments={},
            serial=len(self.reports),
        )

        self.reports[analysis.serial] = analysis
        return analysis
