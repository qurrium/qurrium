"""EchoListenRandomizedV1 - Qurrium
(:mod:`qurry.qurrech.randomized_measure_v1.qurry`)

This is a deprecated version of the randomized measure module.

"""

from pathlib import Path
from typing import Union, Optional, Type, Literal, Iterable
from collections.abc import Hashable
import tqdm

from qiskit import QuantumCircuit
from qiskit.providers import Backend

from .arguments import (
    SHORT_NAME,
    EchoListenRandomizedV1MeasureArgs,
    EchoListenRandomizedV1OutputArgs,
    EchoListenRandomizedV1AnalyzeArgs,
)
from .experiment import (
    EchoListenRandomizedV1Experiment,
    PostProcessingBackendLabel,
    DEFAULT_PROCESS_BACKEND,
)
from ...qurrium import QurriumPrototype
from ...declare import RunArgsType, TranspileArgs, PassManagerType, SpecificAnalsisArgs


class EchoListenRandomizedV1(
    QurriumPrototype[
        EchoListenRandomizedV1Experiment,
        EchoListenRandomizedV1MeasureArgs,
        EchoListenRandomizedV1OutputArgs,
        EchoListenRandomizedV1AnalyzeArgs,
    ]
):
    """Randomized Measure for wave function overlap.
    a.k.a. loschmidt echo when processes time evolution system.

    Reference:
        .. note::
            - Statistical correlations between locally randomized measurements:
            A toolbox for probing entanglement in many-body quantum states -
            A. Elben, B. Vermersch, C. F. Roos, and P. Zoller,
            [PhysRevA.99.052323](
                https://doi.org/10.1103/PhysRevA.99.052323
            )

        .. code-block:: bibtex
            @article{PhysRevA.99.052323,
                title = {Statistical correlations between locally randomized measurements:
                A toolbox for probing entanglement in many-body quantum states},
                author = {Elben, A. and Vermersch, B. and Roos, C. F. and Zoller, P.},
                journal = {Phys. Rev. A},
                volume = {99},
                issue = {5},
                pages = {052323},
                numpages = {12},
                year = {2019},
                month = {May},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevA.99.052323},
                url = {https://link.aps.org/doi/10.1103/PhysRevA.99.052323}
            }
    """

    __name__ = "EchoListenRandomizedV1"
    short_name = SHORT_NAME

    @property
    def experiment_instance(self) -> Type[EchoListenRandomizedV1Experiment]:
        """The container class responding to this Qurrium class."""
        return EchoListenRandomizedV1Experiment

    def measure_to_output(
        self,
        wave1: Optional[Union[QuantumCircuit, Hashable]] = None,
        wave2: Optional[Union[QuantumCircuit, Hashable]] = None,
        times: int = 100,
        measure: Union[int, tuple[int, int], None] = None,
        unitary_loc: Union[int, tuple[int, int], None] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
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
    ) -> EchoListenRandomizedV1OutputArgs:
        """Trasnform :meth:`measure` arguments form into :meth:`output` form.

        Args:
            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            times (int, optional):
                The number of random unitary operator.
                It will denote as `N_U` in the experiment name.
                Defaults to `100`.
            measure (Union[int, tuple[int, int], None], optional):
                The measure range. Defaults to None.
            unitary_loc (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator. Defaults to None.
            random_unitary_seeds (Optional[dict[int, dict[int, int]]], optional):
                The seeds for all random unitary operator.
                This argument only takes input as type of `dict[int, dict[int, int]]`.
                The first key is the index for the random unitary operator.
                The second key is the index for the qubit.

                .. code-block:: python
                    {
                        0: {0: 1234, 1: 5678},
                        1: {0: 2345, 1: 6789},
                        2: {0: 3456, 1: 7890},
                    }

                If you want to generate the seeds for all random unitary operator,
                you can use the function `generate_random_unitary_seeds`
                in `qurry.qurrium.utils.random_unitary`.

                .. code-block:: python
                    from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds
                    random_unitary_seeds = generate_random_unitary_seeds(100, 2)
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
            EchoListenRandomizedOutputArgs: The output arguments.
        """
        if wave1 is None:
            raise ValueError("The `wave` must be provided.")
        if wave2 is None:
            raise ValueError("The `wave2` must be provided.")

        return {
            "circuits": [wave1, wave2],
            "times": times,
            "measure": measure,
            "unitary_loc": unitary_loc,
            "random_unitary_seeds": random_unitary_seeds,
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
        times: int = 100,
        measure: Union[int, tuple[int, int], None] = None,
        unitary_loc: Union[int, tuple[int, int], None] = None,
        random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
        # basic inputs
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
            wave1 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            wave2 (Union[QuantumCircuit, Hashable]):
                The key or the circuit to execute.
            times (int, optional):
                The number of random unitary operator.
                It will denote as `N_U` in the experiment name.
                Defaults to `100`.
            measure (Union[int, tuple[int, int], None], optional):
                The measure range. Defaults to None.
            unitary_loc (Union[int, tuple[int, int], None], optional):
                The range of the unitary operator. Defaults to None.
            random_unitary_seeds (Optional[dict[int, dict[int, int]]], optional):
                The seeds for all random unitary operator.
                This argument only takes input as type of `dict[int, dict[int, int]]`.
                The first key is the index for the random unitary operator.
                The second key is the index for the qubit.

                .. code-block:: python
                    {
                        0: {0: 1234, 1: 5678},
                        1: {0: 2345, 1: 6789},
                        2: {0: 3456, 1: 7890},
                    }

                If you want to generate the seeds for all random unitary operator,
                you can use the function `generate_random_unitary_seeds`
                in `qurry.qurrium.utils.random_unitary`.

                .. code-block:: python
                    from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds
                    random_unitary_seeds = generate_random_unitary_seeds(100, 2)
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
            times=times,
            measure=measure,
            unitary_loc=unitary_loc,
            random_unitary_seeds=random_unitary_seeds,
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

    def multiAnalysis(
        self,
        summoner_id: str,
        *,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: SpecificAnalsisArgs[EchoListenRandomizedV1AnalyzeArgs] = None,
        skip_write: bool = False,
        multiprocess_write: bool = False,
        # analysis arguments
        degree: Optional[Union[tuple[int, int], int]] = None,
        counts_used: Optional[Iterable[int]] = None,
        workers_num: Optional[int] = None,
        backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
        **analysis_args,
    ) -> str:
        """Run the analysis for multiple experiments.

        Args:
            summoner_id (str): The summoner_id of multimanager.
            analysis_name (str, optional):
                The name of analysis. Defaults to 'report'.
            no_serialize (bool, optional):
                Whether to serialize the analysis. Defaults to False.
            specific_analysis_args (
                SpecificAnalsisArgs[EchoListenRandomizedV1AnalyzeArgs], optional
            ):
                The specific arguments for analysis. Defaults to None.
            compress (bool, optional):
                Whether to compress the export file. Defaults to False.
            skip_write (bool, optional):
                Whether to skip the file writing during the analysis. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

            degree (Union[tuple[int, int], int]): Degree of the subsystem.
            counts_used (Optional[Iterable[int]], optional):
                The index of the counts used.
                If not specified, then use all counts.
                Defaults to None.
            workers_num (Optional[int], optional):
                Number of multi-processing workers,
                if sets to 1, then disable to using multi-processing;
                if not specified, then use the number of all cpu counts - 2 by `cpu_count() - 2`.
                Defaults to None.
            backend (PostProcessingBackendLabel, optional):
                Backend for the process. Defaults to DEFAULT_PROCESS_BACKEND.

        Returns:
            str: The summoner_id of multimanager.
        """

        return super().multiAnalysis(
            summoner_id=summoner_id,
            analysis_name=analysis_name,
            no_serialize=no_serialize,
            specific_analysis_args=specific_analysis_args,
            skip_write=skip_write,
            multiprocess_write=multiprocess_write,
            degree=degree,
            counts_used=counts_used,
            workers_num=workers_num,
            backend=backend,
            **analysis_args,
        )
