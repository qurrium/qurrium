"""MagnetSquare - Experiment (:mod:`qurry.qurries.magnet_square.experiment`)"""

from collections.abc import Hashable
from typing import Optional, Type, Any
from itertools import permutations
import tqdm

from qiskit import QuantumCircuit

from .analysis import MagnetSquareAnalysis
from .arguments import MagnetSquareArguments, SHORT_NAME
from .utils import circuit_method

from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...process.magnet_square.magnet_square import (
    magnet_square,
    MagnetSquare,
    DEFAULT_PROCESS_BACKEND,
    PostProcessingBackendLabel,
)
from ...tools import ParallelManager, set_pbar_description


class MagnetSquareExperiment(
    ExperimentPrototype[
        MagnetSquareArguments,
        MagnetSquareAnalysis,
    ]
):
    """The instance of experiment."""

    __name__ = "EntropyMeasureRandomizedExperiment"

    @property
    def arguments_instance(self) -> Type[MagnetSquareArguments]:
        """The arguments instance for this experiment."""
        return MagnetSquareArguments

    @property
    def analysis_instance(self) -> Type[MagnetSquareAnalysis]:
        """The analysis instance for this experiment."""
        return MagnetSquareAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        **custom_kwargs: Any,
    ) -> tuple[MagnetSquareArguments, Commonparams, dict[str, Any]]:
        """Handling all arguments and initializing a single experiment.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            exp_name (str, optional):
                The name of the experiment.
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.
            custom_kwargs (Any):
                The custom parameters.

        Returns:
            tuple[MagnetSquareArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) > 1:
            raise ValueError("The number of target circuits should be only one.")

        target_key, target_circuit = targets[0]
        actual_qubits = target_circuit.num_qubits

        exp_name = f"{exp_name}.{SHORT_NAME}"

        # pylint: disable=protected-access
        return MagnetSquareArguments._filter(
            exp_name=exp_name,
            target_keys=[target_key],
            num_qubits=actual_qubits,
            **custom_kwargs,
        )
        # pylint: enable=protected-access

    @classmethod
    def method(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        arguments: MagnetSquareArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (MagnetSquareArguments):
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
        permut = permutations(range(arguments.num_qubits), 2)
        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)

        set_pbar_description(pbar, "Building circuits...")
        if multiprocess:
            pool = ParallelManager()
            circ_list = pool.starmap(
                circuit_method,
                [
                    (k, target_circuit, target_key, arguments.exp_name, i, j)
                    for k, (i, j) in enumerate(permut)
                ],
            )
        else:
            circ_list = [
                circuit_method(k, target_circuit, target_key, arguments.exp_name, i, j)
                for k, (i, j) in enumerate(permut)
            ]

        return circ_list, {}

    def analyze(
        self,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> MagnetSquareAnalysis:
        """Calculate magnet square with more information combined.

        Args:
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to `None`.

        Returns:
            MagnetSquareAnalysis: The result of the analysis.
        """

        qs = self.quantities(
            shots=self.commons.shots,
            counts=self.afterwards.counts,
            num_qubits=self.args.num_qubits,
            pbar=pbar,
        )

        serial = len(self.reports)
        analysis = self.analysis_instance(
            serial=serial,
            shots=self.commons.shots,
            **qs,
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        counts: Optional[list[dict[str, int]]] = None,
        num_qubits: Optional[int] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> MagnetSquare:
        """Calculate magnet square with more information combined.

        Args:
            shots (int):
                The number of shots.
            counts (list[dict[str, int]]):
                The counts of the experiment.
            num_qubits (int):
                The number of qubits.
            backend (PostProcessingBackendLabel, optional):
                The backend label. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to `None`.

        Returns:
            MagnetSquare: The result of the magnet square.
        """

        if counts is None:
            raise ValueError("The counts should be given.")
        if num_qubits is None:
            raise ValueError("The number of qubits should be given.")
        if shots is None:
            raise ValueError("The number of shots should be given.")

        return magnet_square(
            shots=shots,
            counts=counts,
            num_qubits=num_qubits,
            backend=backend,
            pbar=pbar,
        )
