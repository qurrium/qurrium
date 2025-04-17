"""Declaration - Run - Simulator (:mod:`qurry.declare.run.simulator`)

This module reveals the full arguments of the :meth:`backend.run` method
for the simulator backends to avoid the UNnEcEsSaRy and PAINFUL finding of
how many arguments and what types of arguments you can pass to the method.

"""

from typing import Optional, Any

from qiskit.circuit import Parameter

from .base_run import BaseRunArgs


class BasicSimulatorRunArgs(BaseRunArgs, total=False):
    """Arguments for :meth:`backend.run` from :mod:`qiskit.providers.backend`.
    For :cls:`BasicSimulator` from :mod:`qiskit.providers.basic_provider`:

    - For qiskit>=2.0, the signature of :meth:`backend.run` is:

    .. code-block:: python
        def run(
            self, run_input: QuantumCircuit | list[QuantumCircuit], **run_options
        ) -> BasicProviderJob:

    ->

    .. code-block:: python
        @classmethod
        def _default_options(cls) -> Options:
            return Options(
                shots=1024,
                memory=True,
                initial_statevector=None,
                seed_simulator=None,
            )

    - For qiskit<2.0, the signature of :meth:`backend.run` is:
    .. code-block:: python
        def run(
            self, run_input: QuantumCircuit | list[QuantumCircuit], **backend_options
        ) -> BasicProviderJob:

    ->

    .. code-block:: python
        @classmethod
        def _default_options(cls) -> Options:
            return Options(
                shots=1024,
                memory=False,
                initial_statevector=None,
                chop_threshold=1e-15,
                allow_sample_measuring=True,
                seed_simulator=None,
                parameter_binds=None,
            )

    or ?

    .. code-block:: python
        def _assemble(
            experiments: Union[
                QuantumCircuit,
                List[QuantumCircuit],
                Schedule,
                List[Schedule],
                ScheduleBlock,
                List[ScheduleBlock],
            ],
            backend: Optional[Backend] = None,
            qobj_id: Optional[str] = None,
            qobj_header: Optional[Union[QobjHeader, Dict]] = None,
            shots: Optional[int] = None,
            memory: Optional[bool] = False,
            seed_simulator: Optional[int] = None,
            qubit_lo_freq: Optional[List[float]] = None,
            meas_lo_freq: Optional[List[float]] = None,
            qubit_lo_range: Optional[List[float]] = None,
            meas_lo_range: Optional[List[float]] = None,
            schedule_los: Optional[
                Union[
                    List[Union[Dict[PulseChannel, float], LoConfig]],
                    Union[Dict[PulseChannel, float], LoConfig],
                ]
            ] = None,
            meas_level: Union[int, MeasLevel] = MeasLevel.CLASSIFIED,
            meas_return: Union[str, MeasReturnType] = MeasReturnType.AVERAGE,
            meas_map: Optional[List[List[Qubit]]] = None,
            memory_slot_size: int = 100,
            rep_time: Optional[int] = None,
            rep_delay: Optional[float] = None,
            parameter_binds: Optional[List[Dict[Parameter, float]]] = None,
            parametric_pulses: Optional[List[str]] = None,
            init_qubits: bool = True,
            **run_config: Dict,
        ) -> Union[QasmQobj, PulseQobj]:
        ...

    """

    shots: Optional[int]
    memory: Optional[bool]
    initial_statevector: Optional[Any]
    seed_simulator: Optional[int]
    chop_threshold: Optional[float]
    allow_sample_measuring: Optional[bool]
    parameter_binds: Optional[list[dict[Parameter, float]]]


class AerBackendRunArgs(BaseRunArgs, total=False):
    """Arguments for :meth:`backend.run` from :mod:`qiskit.providers.backend`.
    For :cls:`AerBackend` from :mod:`qiskit_aer.backends.aerbackend`
    or :cls:`AerBackend` from :mod:`qiskit.providers.aer.backends.aerbackend`,
    the old import path.:

    .. code-block:: python
        def run(self, circuits, parameter_binds=None, **run_options):
            if isinstance(circuits, (QuantumCircuit, Schedule, ScheduleBlock)):
            circuits = [circuits]

            return self._run_circuits(circuits, parameter_binds, **run_options)

    ->

    .. code-block:: python
        def _run_circuits(self, circuits, parameter_binds, **run_options):
            # Submit job
            job_id = str(uuid.uuid4())
            aer_job = AerJob(
                self,
                job_id,
                self._execute_circuits_job, # This takes run_options
                parameter_binds=parameter_binds,
                circuits=circuits,
                run_options=run_options,
            )
            aer_job.submit()

            return aer_job

    ->

    .. code-block:: python
        def set_option(self, key, value):
            if hasattr(self._configuration, key):
                self._set_configuration_option(key, value)
            elif hasattr(self._properties, key):
                self._set_properties_option(key, value)
            else:
                if not hasattr(self._options, key):
                    raise AerError(f"Invalid option {key}")
                if value is not None:
                    # Only add an option if its value is not None
                    setattr(self._options, key, value)
                else:
                    # If setting an existing option to None reset it to default
                    # this is for backwards compatibility when setting it to None would
                    # remove it from the options dict
                    setattr(self._options, key, getattr(self._default_options(), key))

    `run_options` is a dictionary that will temporarily override any set options
    According to the default options has been set in the backend,
    the default options are:

    .. code-block:: python
        @classmethod
        def _default_options(cls):
            return Options(
                # Global options
                shots=1024,
                method="automatic",
                device="CPU",
                precision="double",
                executor=None,
                max_job_size=None,
                max_shot_size=None,
                enable_truncation=True,
                zero_threshold=1e-10,
                validation_threshold=None,
                max_parallel_threads=None,
                max_parallel_experiments=None,
                max_parallel_shots=None,
                max_memory_mb=None,
                fusion_enable=True,
                fusion_verbose=False,
                fusion_max_qubit=None,
                fusion_threshold=None,
                accept_distributed_results=None,
                memory=None,
                noise_model=None,
                seed_simulator=None,
                # cuStateVec (cuQuantum) option
                cuStateVec_enable=False,
                # cache blocking for multi-GPUs/MPI options
                blocking_qubits=None,
                blocking_enable=False,
                chunk_swap_buffer_qubits=None,
                # multi-shots optimization options (GPU only)
                batched_shots_gpu=False,
                batched_shots_gpu_max_qubits=16,
                num_threads_per_device=1,
                # multi-shot branching
                shot_branching_enable=False,
                shot_branching_sampling_enable=False,
                # statevector options
                statevector_parallel_threshold=14,
                statevector_sample_measure_opt=10,
                # stabilizer options
                stabilizer_max_snapshot_probabilities=32,
                # extended stabilizer options
                extended_stabilizer_sampling_method="resampled_metropolis",
                extended_stabilizer_metropolis_mixing_time=5000,
                extended_stabilizer_approximation_error=0.05,
                extended_stabilizer_norm_estimation_samples=100,
                extended_stabilizer_norm_estimation_repetitions=3,
                extended_stabilizer_parallel_threshold=100,
                extended_stabilizer_probabilities_snapshot_samples=3000,
                # MPS options
                matrix_product_state_truncation_threshold=1e-16,
                matrix_product_state_max_bond_dimension=None,
                mps_sample_measure_algorithm="mps_heuristic",
                mps_log_data=False,
                mps_swap_direction="mps_swap_left",
                chop_threshold=1e-8,
                mps_parallel_threshold=14,
                mps_omp_threads=1,
                mps_lapack=False,
                # tensor network options
                tensor_network_num_sampling_qubits=10,
                use_cuTensorNet_autotuning=False,
                # parameter binding
                runtime_parameter_bind_enable=False,
            )

    (Captured from qiskit-aer 0.16.0)
    """

    parameter_binds: Optional[list[dict[Parameter, float]]]

    shots: Optional[int]
    method: Optional[str]
    device: Optional[str]
    precision: Optional[str]
    executor: Optional[str]
    max_job_size: Optional[int]
    max_shot_size: Optional[int]
    enable_truncation: Optional[bool]
    zero_threshold: Optional[float]
    validation_threshold: Optional[float]
    max_parallel_threads: Optional[int]
    max_parallel_experiments: Optional[int]
    max_parallel_shots: Optional[int]
    max_memory_mb: Optional[int]
    fusion_enable: Optional[bool]
    fusion_verbose: Optional[bool]
    fusion_max_qubit: Optional[int]
    fusion_threshold: Optional[float]
    accept_distributed_results: Optional[bool]
    memory: Optional[bool]
    noise_model: Optional[Any]
    seed_simulator: Optional[int]
    # cuStateVec (cuQuantum) option
    cuStateVec_enable: Optional[bool]
    # cache blocking for multi-GPUs/MPI options
    blocking_qubits: Optional[list[int]]
    blocking_enable: Optional[bool]
    chunk_swap_buffer_qubits: Optional[list[int]]
    # multi-shots optimization options (GPU only)
    batched_shots_gpu: Optional[bool]
    batched_shots_gpu_max_qubits: Optional[int]
    num_threads_per_device: Optional[int]
    # multi-shot branching
    shot_branching_enable: Optional[bool]
    shot_branching_sampling_enable: Optional[bool]
    # statevector options
    statevector_parallel_threshold: Optional[int]
    statevector_sample_measure_opt: Optional[int]
    # stabilizer options
    stabilizer_max_snapshot_probabilities: Optional[int]
    # extended stabilizer options
    extended_stabilizer_sampling_method: Optional[str]
    extended_stabilizer_metropolis_mixing_time: Optional[int]
    extended_stabilizer_approximation_error: Optional[float]
    extended_stabilizer_norm_estimation_samples: Optional[int]
    extended_stabilizer_norm_estimation_repetitions: Optional[int]
    extended_stabilizer_parallel_threshold: Optional[int]
    extended_stabilizer_probabilities_snapshot_samples: Optional[int]
    # MPS options
    matrix_product_state_truncation_threshold: Optional[float]
    matrix_product_state_max_bond_dimension: Optional[int]
    mps_sample_measure_algorithm: Optional[str]
    mps_log_data: Optional[bool]
    mps_swap_direction: Optional[str]
    chop_threshold: Optional[float]
    mps_parallel_threshold: Optional[int]
    mps_omp_threads: Optional[int]
    mps_lapack: Optional[bool]
    # tensor network options
    tensor_network_num_sampling_qubits: Optional[int]
    use_cuTensorNet_autotuning: Optional[bool]
    # parameter binding
    runtime_parameter_bind_enable: Optional[bool]


class BasicAerBackendRunArgs(BaseRunArgs, total=False):
    """Arguments for :meth:`backend.run` from :mod:`qiskit.providers.backend`.
    For :cls:`QasmSimulatorPy` from :mod:`qiskit.providers.basicaer`:

    .. code-block:: python
        def run(self, qobj, **backend_options):
            ...
            self._set_options(qobj_config=qobj_options, backend_options=backend_options)
            job_id = str(uuid.uuid4())
            job = BasicAerJob(self, job_id, self._run_job(job_id, qobj))
            return job

    ->

    ???

    Well, It's an example of a FAILED finding
    Also, this simulator is just lived short in qiskit then deprecated.
    So, I don't think it's worth to find the arguments for this simulator anymore.
    """
