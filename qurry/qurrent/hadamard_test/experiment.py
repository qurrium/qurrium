"""EntropyMeasureHadamard - Experiment (:mod:`qurry.qurrent.hadamard_test.experiment`)"""

from typing import Optional, Any
import tqdm

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from .analysis import EMHAnalysis
from .arguments import EMHArguments, SHORT_NAME
from ...qurrium import ExperimentPrototype, Commonparams, WCKeyable
from ...process.utils import qubit_selector


class EMHExperiment(ExperimentPrototype[EMHArguments, EMHAnalysis]):
    """The instance of experiment."""

    __name__ = "EMHExperiment"

    @classmethod
    def arguments_type(cls) -> type[EMHArguments]:
        """The arguments instance for this experiment."""
        return EMHArguments

    @classmethod
    def analysis_type(cls) -> type[EMHAnalysis]:
        """The analysis instance for this experiment."""
        return EMHAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[WCKeyable, QuantumCircuit]],
        exp_name: str = "exps",
        degree: Optional[tuple[int, int]] = None,
        **custom_kwargs: Any,
    ) -> tuple[EMHArguments, Commonparams, dict[str, Any]]:
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
            ValueError: The number of target circuits should be 1.

        Returns:
            The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) != 1:
            raise ValueError("The number of target circuits should be 1.")

        target_key, target_circuit = targets[0]
        num_qubits = target_circuit.num_qubits
        degree = qubit_selector(num_qubits, degree=degree)

        exp_name = f"{exp_name}.degree_{degree[0]}_{degree[1]}.{SHORT_NAME}"

        return EMHArguments.filter(
            exp_name=exp_name,
            target_keys=[target_key],
            degree=degree,
            **custom_kwargs,
        )

    @classmethod
    def method(
        cls,
        targets: list[tuple[WCKeyable, QuantumCircuit]],
        arguments: EMHArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = False,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[WCKeyable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (EntropyMeasureHadamardArguments):
                The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the arguments of the experiment.
        """

        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)
        num_qubits = target_circuit.num_qubits
        old_name = "" if isinstance(target_circuit.name, str) else target_circuit.name

        q_ancilla = QuantumRegister(1, "ancilla_1")
        q_func1 = QuantumRegister(num_qubits, "q1")
        q_func2 = QuantumRegister(num_qubits, "q2")
        c_meas1 = ClassicalRegister(1, "c1")
        qc_exp1 = QuantumCircuit(q_ancilla, q_func1, q_func2, c_meas1)
        qc_exp1.name = (
            f"{arguments.exp_name}" + ""
            if len(target_key) < 1
            else f".{target_key}" + ""
            if len(old_name) < 1
            else f".{old_name}"
        )

        qc_exp1.compose(
            target_circuit,
            [q_func1[i] for i in range(num_qubits)],
            inplace=True,
        )

        qc_exp1.compose(
            target_circuit,
            [q_func2[i] for i in range(num_qubits)],
            inplace=True,
        )

        qc_exp1.barrier()
        qc_exp1.h(q_ancilla)
        for i in range(*arguments.degree):
            qc_exp1.cswap(q_ancilla[0], q_func1[i], q_func2[i])
        qc_exp1.h(q_ancilla)
        qc_exp1.measure(q_ancilla, c_meas1)

        return [qc_exp1], {}

    def analyze(self, pbar: Optional[tqdm.tqdm] = None) -> EMHAnalysis:
        """Calculate entangled entropy with more information combined.

        Args:
            degree (Union[tuple[int, int], int]): Degree of the subsystem.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            EntropyMeasureHadamardAnalysis: The result of the analysis.
        """

        if pbar is not None:
            pbar.set_description("Calculating entangled entropy")

        analysis = self.analysis_type().perform_analysis(
            arguments=self.args,
            commonparams=self.commons,
            counts=self.afterwards.counts,
            analyze_arguments={},
            serial=len(self.reports),
        )

        self.reports[analysis.serial] = analysis
        return self.reports[analysis.serial]
