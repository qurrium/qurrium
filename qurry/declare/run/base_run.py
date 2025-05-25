"""Declaration - Run - BaseRunArgs (:mod:`qurry.declare.run.base_run`)"""

from typing import Optional, Union, TypedDict, Any


class BaseRunArgs(TypedDict):
    """Arguments for :meth:`run` of :cls:`Backend` from :mod:`qiskit.providers.backend` ."""


RunArgsType = Optional[Union[BaseRunArgs, dict[str, Any]]]
"""The type hint for run arguments in :meth:`output` from :cls:`QurriumPrototype`."""
