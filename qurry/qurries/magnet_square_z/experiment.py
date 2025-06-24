"""ZDirMagnetSquare - Experiment (:mod:`qurry.qurries.magnet_square_z.experiment`)"""

from collections.abc import Hashable
from typing import Optional, Type, Any
import tqdm

from qiskit import QuantumCircuit

from .analysis import ZDirMagnetSquareAnalysis
from .arguments import ZDirMagnetSquareArguments, SHORT_NAME
from .utils import circuit_method

from ...qurrium.experiment import ExperimentPrototype, Commonparams
from ...process.magnet_square import z_dir_magnet_square, MagnetSquare, DEFAULT_PROCESS_BACKEND
from ...process.availability import PostProcessingBackendLabel


class ZDirMagnetSquareExperiment(
    ExperimentPrototype[ZDirMagnetSquareArguments, ZDirMagnetSquareAnalysis]
):
    """The instance of experiment."""

    __name__ = "ZDirMagnetSquareExperiment"

    @property
    def arguments_instance(self) -> Type[ZDirMagnetSquareArguments]:
        """The arguments instance for this experiment."""
        return ZDirMagnetSquareArguments

    @property
    def analysis_instance(self) -> Type[ZDirMagnetSquareAnalysis]:
        """The analysis instance for this experiment."""
        return ZDirMagnetSquareAnalysis

    @classmethod
    def params_control(
        cls,
        targets: list[tuple[Hashable, QuantumCircuit]],
        exp_name: str = "exps",
        **custom_kwargs: Any,
    ) -> tuple[ZDirMagnetSquareArguments, Commonparams, dict[str, Any]]:
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
            tuple[ZDirMagnetSquareArguments, Commonparams, dict[str, Any]]:
                The arguments of the experiment, the common parameters, and the custom parameters.
        """
        if len(targets) > 1:
            raise ValueError("The number of target circuits should be only one.")

        target_key, target_circuit = targets[0]
        actual_qubits = target_circuit.num_qubits

        exp_name = f"{exp_name}.{SHORT_NAME}"

        # pylint: disable=protected-access
        return ZDirMagnetSquareArguments._filter(
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
        arguments: ZDirMagnetSquareArguments,
        pbar: Optional[tqdm.tqdm] = None,
        multiprocess: bool = True,
    ) -> tuple[list[QuantumCircuit], dict[str, Any]]:
        """The method to construct circuit.

        Args:
            targets (list[tuple[Hashable, QuantumCircuit]]):
                The circuits of the experiment.
            arguments (ZDirMagnetSquareArguments):
                The arguments of the experiment.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for showing the progress of the experiment. Defaults to None.
            multiprocess (bool, optional):
                Whether to use multiprocessing. Defaults to `True`.

        Returns:
            tuple[list[QuantumCircuit], dict[str, Any]]:
                The circuits of the experiment and the side products.
        """

        target_key, target_circuit = targets[0]
        target_key = "" if isinstance(target_key, int) else str(target_key)
        return [circuit_method(target_circuit, target_key, arguments.exp_name)], {}

    def analyze(self, pbar: Optional[tqdm.tqdm] = None) -> ZDirMagnetSquareAnalysis:
        """Calculate magnet square with more information combined.

        Args:
            pbar (Optional[tqdm.tqdm], optional): The progress bar. Defaults to None.

        Returns:
            ZDirMagnetSquareAnalysis: The result of the magnet square analysis.
        """

        assert (
            len(self.afterwards.counts) == 1
        ), f"The number of counts should be one, but got {len(self.afterwards.counts)}."

        qs = self.quantities(
            shots=self.commons.shots,
            single_counts=self.afterwards.counts[0],
            num_qubits=self.args.num_qubits,
            pbar=pbar,
        )

        serial = len(self.reports)
        analysis = self.analysis_instance(
            serial=serial, shots=self.commons.shots, num_qubits=self.args.num_qubits, **qs
        )

        self.reports[serial] = analysis
        return analysis

    @classmethod
    def quantities(
        cls,
        shots: Optional[int] = None,
        single_counts: Optional[dict[str, int]] = None,
        num_qubits: Optional[int] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> MagnetSquare:
        """Calculate magnet square with more information combined.

        Args:
            shots (int):
                The number of shots.
            single_counts (dict[str, int]): Single count.
            num_qubits (int):
                The number of qubits.
            backend (PostProcessingBackendLabel, optional):
                The backend label. Defaults to DEFAULT_PROCESS_BACKEND.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar. Defaults to None.

        Returns:
            MagnetSquare: The result of the magnet square.
        """

        if single_counts is None:
            raise ValueError("The single_counts should be given.")
        if num_qubits is None:
            raise ValueError("The number of qubits should be given.")
        if shots is None:
            raise ValueError("The number of shots should be given.")

        return z_dir_magnet_square(
            shots=shots,
            single_counts=single_counts,
            num_qubits=num_qubits,
            backend=backend,
            pbar=pbar,
        )
