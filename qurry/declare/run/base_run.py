"""Declaration - Run - BaseRunArgs (:mod:`qurry.declare.run.base_run`)"""

from typing import Optional, Union, TypedDict, Any


class BaseRunArgs(TypedDict):
    """Arguments for :meth:`~qiskit.providers.backend.BackendV2.run`."""


RunArgsType = Optional[Union[BaseRunArgs, dict[str, Any]]]
"""The type hint for :meth:`~qiskit.providers.backend.BackendV2.run`."""
