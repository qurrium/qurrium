"""StringOperator - Experiment (:mod:`qurry.qurries.string_operator.experiment`)"""

from collections.abc import Hashable
from typing import Optional, Type, Any
import tqdm

from qiskit import QuantumCircuit

from .analysis import StringOperatorAnalysis
from .arguments import StringOperatorArguments, SHORT_NAME
from .utils import circuit_method, StringOperatorLibType, StringOperatorDirection, STRING_OPERATOR

from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...process.string_operator.string_operator import (
    string_operator_order,
    StringOperator,
    DEFAULT_PROCESS_BACKEND,
    PostProcessingBackendLabel,
)
from ...tools import set_pbar_description


class StringOperatorExperiment(
    ExperimentPrototype[
        StringOperatorArguments,
        StringOperatorAnalysis,
    ]
):
    """The instance of experiment."""

    __name__ = "EntropyMeasureRandomizedExperiment"
    short_name = SHORT_NAME

    @property
    def arguments_instance(self) -> Type[StringOperatorArguments]:
        """The arguments instance for this experiment."""
        return StringOperatorArguments

    @property
    def analysis_instance(self) -> Type[StringOperatorAnalysis]:
        """The analysis instance for this experiment."""
        return StringOperatorAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        i: Optional[int] = None,
        k: Optional[int] = None,
        str_op: StringOperatorLibType = "i",
        on_dir: StringOperatorDirection = "x",
        **custom_kwargs: Any,
    ) -> tuple[StringOperatorArguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            i (Optional[int], optional):
                The index of beginning qubits in the quantum circuit.
            k (Optional[int], optional):
                The index of ending qubits in the quantum circuit.
            str_op (StringOperatorLibType, optional):
                The string operator. Defaults to "i".
            on_dir (StringOperatorDirection, optional):
                The direction of the string operator, either 'x' or 'y'. Defaults to "x".
            custom_kwargs (Any):
                The custom parameters.

        Returns:
            tuple[StringOperatorArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) > 1:
            raise ValueError("The number of target circuits should be only one.")
        target_key, target_circuit = targets[0]
        num_qubits = target_circuit.num_qubits

        if on_dir not in STRING_OPERATOR:
            raise ValueError("The `on_dir` must be either 'x' or 'y'.")
        if str_op not in STRING_OPERATOR[on_dir]:
            raise ValueError(f"The `str_op` must be one of {list(STRING_OPERATOR[on_dir])}.")

        if k is None:
            k = num_qubits - 1
        if i is None:
            i = 0
        if i >= k:
            raise ValueError(f"i: {i} is not less than k: {k}.")

        if k - i + 1 < len(STRING_OPERATOR[on_dir][str_op]):
            raise ValueError(
                f"The `k - i + 1` must be greater than or equal to "
                f"{len(STRING_OPERATOR[on_dir][str_op])}. But got k: {k} - i: {i} = {k - i + 1}."
            )

        # pylint: disable=protected-access
        return StringOperatorArguments._filter(
            exp_name=f"{exp_name}.i_{i}_k_{k}.op_{str_op}_dir_{on_dir}.{SHORT_NAME}",
            target_keys=[target_key],
            num_qubits=num_qubits,
            str_op=str_op,
            i=i,
            k=k,
            **custom_kwargs,
        )
        # pylint: enable=protected-access

    @classmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: StringOperatorArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (StringOperatorArguments):
                The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the side products.
        """
        set_pbar_description(pbar, f"Prepare permutation for {arguments.num_qubits} qubits.")
        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)

        assert arguments.i is not None and arguments.k is not None, (
            f"i and k should be given, but got {arguments.i} and {arguments.k}. "
            "Please check the arguments."
        )

        return [
            circuit_method(
                target_circuit,
                target_key,
                arguments.i,
                arguments.k,
                arguments.str_op,
                arguments.on_dir,
            )
        ], {}

    def analyze(self, pbar: Optional[tqdm.tqdm] = None) -> StringOperatorAnalysis:
        """Calculate magnet square with more information combined.

        Args:
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            StringOperatorAnalysis: The result of the analysis.
        """

        qs = self.quantities(shots=self.commons.shots, counts=self.afterwards.counts, pbar=pbar)

        serial = len(self.reports)
        analysis = self.analysis_instance(
            i=self.args.i,
            k=self.args.k,
            length=self.args.k - self.args.i + 1,
            str_op=self.args.str_op,
            on_dir=self.args.on_dir,
            num_qubits=self.args.num_qubits,
            shots=self.commons.shots,
            serial=serial,
            **qs,
        )
        assert analysis.content.k - analysis.content.i + 1 == analysis.content.length, (
            f"Length of the string operator should be equal to k - i + 1, "
            f"but got length: {analysis.content.length} != "
            f"k - i + 1: {analysis.content.k - analysis.content.i + 1}."
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        counts: Optional[list[dict[str, int]]] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> StringOperator:
        """Calculate the string operator.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The counts of the experiment.
            backend (PostProcessingBackendLabel, optional):
                The backend label. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            StringOperator: The result of the magnet square.
        """

        if shots is None:
            raise ValueError("The number of shots should be given.")
        if counts is None:
            raise ValueError("The counts should be given.")

        return string_operator_order(
            shots=shots,
            counts=counts,
            backend=backend,
            pbar=pbar,
        )
