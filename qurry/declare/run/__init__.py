"""Declaration - Run (:mod:`qurry.declare.run`)

Arguments for :meth:`~qiskit.providers.backend.BackendV2.run`
of :class:`~qiskit.providers.backend.BackendV2` from :mod:`~qiskit.providers.backend`
"""

from .base_run import BaseRunArgs, RunArgsType
from .ibm import IBMRuntimeBackendRunArgs, IBMProviderBackendRunArgs, IBMQBackendRunArgs
from .simulator import BasicSimulatorRunArgs, AerBackendRunArgs, BasicAerBackendRunArgs
