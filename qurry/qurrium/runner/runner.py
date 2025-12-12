"""Runner for pending and retrieve jobs from remote backend.
(:mod:`qurry.qurrium.runner.runner`)
"""

import warnings
from abc import abstractmethod, ABC
from typing import Optional, Literal

from qiskit.providers import Backend
from qiskit.result import Result

from ..multimanager import MultiManager
from ..multimanager.beforewards import TagsType
from ..exceptions import QurryDummyRunnerWarning
from ...tools.backend import backend_name_getter


class Runner(ABC):
    """Pending and Retrieve Jobs from remote backend."""

    __name__ = "Runner"

    current_multimanager: MultiManager
    """The current :class:`~qurry.qurrium.multimanager.multimanager.Multimanager` been used."""
    backend: Optional[Backend]
    """The backend been used."""

    retrieve_results: dict[TagsType, Result]
    """The retrieved results from remote backend. Key is the job id.
    which multiple :class:`~qurry.qurrium.experiment.experiment.ExperimentPrototype` shared.
    """
    all_counts: dict[TagsType, dict[str, int]]
    """The counts of all retrieved results from remote backend. Key is the job id,
    This is used for temporary storage before re-distributing to each experiment.
    """

    @abstractmethod
    def pending(
        self,
        pending_strategy: Literal["default", "onetime", "each", "tags"],
        backend: Backend,
    ) -> list[tuple[Optional[str], TagsType]]:
        """Pending jobs to remote backend."""

    @abstractmethod
    def retrieve(self, overwrite: bool = False) -> list[tuple[Optional[str], TagsType]]:
        """Retrieve jobs from remote backend."""

    def __repr__(self):
        backend_repr = self.backend if self.backend is None else backend_name_getter(self.backend)

        return (
            f"<{self.__name__}("
            + f"current_multimanager={self.current_multimanager._repr_oneline()}, "
            + f"backend={backend_repr}, "
        )

    def _repr_oneline(self):
        backend_repr = self.backend if self.backend is None else backend_name_getter(self.backend)

        return f"<{self.__name__}(backend={backend_repr}, ...)>"


# Using for Third-Party Backend
class ThirdPartyRunner(Runner):
    """Pending and Retrieve Jobs from Third-Parties' backend."""

    @abstractmethod
    def __init__(self, manager: MultiManager, backend: Backend, **kwargs):
        pass


class DummyRunner(Runner):
    """A dummy runner for testing."""

    __name__ = "DummyRunner"

    def __init__(
        self,
        manager: MultiManager,
        backend: Backend,
    ):
        warnings.warn(
            "You are using a dummy runner, which does not support any jobs.",
            QurryDummyRunnerWarning,
        )
        self.current_multimanager = manager
        self.backend = backend
        self.provider = None

        self.reports = {}

    def pending(
        self,
        pending_strategy: Literal["default", "onetime", "each", "tags"],
        backend: Optional[Backend] = None,
    ) -> list[tuple[Optional[str], TagsType]]:
        """ATTENTION!! This method does not do anything.

        Args:
            pending_strategy (Literal["default", "onetime", "each", "tags"]): The pending strategy.
            backend (Optional[Backend], optional): The backend to be used. Defaults to None.

        Returns:
            list[tuple[Optional[str], TagsType]]: The pending jobs.
        """
        warnings.warn(
            "You are using a dummy runner, it does not do anything.",
            QurryDummyRunnerWarning,
        )
        return []

    def retrieve(self, overwrite: bool = False) -> list[tuple[Optional[str], TagsType]]:
        """ATTENTION!! This method does not do anything.

        Args:
            overwrite (bool, optional): Whether to overwrite the jobs. Defaults to False.

        Returns:
            list[tuple[Optional[str], TagsType]]: The retrieved jobs.
        """

        warnings.warn(
            "You are using a dummy runner, it does not do anything.",
            QurryDummyRunnerWarning,
        )
        return []
