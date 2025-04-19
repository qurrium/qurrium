"""StringOperator - Experiment (:mod:`qurry.qurries.string_operator.experiment`)"""

from collections.abc import Hashable
from typing import Optional, Type, Any
import tqdm

from qiskit import QuantumCircuit

from .analysis import StringOperatorAnalysis
from .arguments import StringOperatorArguments, SHORT_NAME
from .utils import circuit_method, AvailableStringOperatorTypes, STRING_OPERATOR_LIB

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
        str_op: AvailableStringOperatorTypes = "i",
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
            str_op (AvailableStringOperatorTypes, optional):
                The string operator.
            custom_kwargs (Any):
                The custom parameters.

        Returns:
            tuple[StringOperatorArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) > 1:
            raise ValueError("The number of target circuits should be only one.")

        if str_op not in STRING_OPERATOR_LIB:
            raise ValueError(
                "The given string is not in the library, "
                + f"please choose from {list(STRING_OPERATOR_LIB.keys())}."
            )

        target_key, target_circuit = targets[0]
        num_qubits = target_circuit.num_qubits
        if num_qubits < len(STRING_OPERATOR_LIB[str_op]):
            raise ValueError(
                f"The given wave function '{target_key}' only has {num_qubits} qubits less than "
                + f"min length {len(STRING_OPERATOR_LIB[str_op])} of string operator {str_op}."
            )

        k = num_qubits - 2 if k is None else k
        i = 0 if i is None else i
        if i >= k:
            raise ValueError(f"'i ({i}) >= k ({k})' which is not allowed")

        length = k - i + 1
        if length < len(STRING_OPERATOR_LIB[str_op]):
            raise ValueError(
                f"The given qubit range i={i} to k={k} only has length={length} less than "
                + f"min length={len(STRING_OPERATOR_LIB[str_op])} of string operator '{str_op}'."
            )

        exp_name = f"{exp_name}.{SHORT_NAME}"

        # pylint: disable=protected-access
        return StringOperatorArguments._filter(
            exp_name=exp_name,
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
                The progress bar for showing the progress of the experiment.
                Defaults to `None`.
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
            f"i and k should be given, but got {arguments.i} and {arguments.k}."
            + "Please check the arguments."
        )

        return [
            circuit_method(
                target_circuit,
                target_key,
                arguments.exp_name,
                arguments.i,
                arguments.k,
                arguments.str_op,
            )
        ], {}

    def analyze(
        self,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> StringOperatorAnalysis:
        """Calculate magnet square with more information combined.

        Args:
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to `None`.

        Returns:
            StringOperatorAnalysis: The result of the analysis.
        """

        qs = self.quantities(
            shots=self.commons.shots,
            counts=self.afterwards.counts,
            pbar=pbar,
        )

        serial = len(self.reports)
        analysis = self.analysis_instance(
            i=self.args.i,
            k=self.args.k,
            str_op=self.args.str_op,
            num_qubits=self.args.num_qubits,
            shots=self.commons.shots,
            serial=serial,
            **qs,
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
                The progress bar. Defaults to `None`.

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
