"""MultiManager - Afterwards (:mod:`qurry.qurrium.multimanager.afterwards`)"""

from collections.abc import Hashable
from typing import NamedTuple

from qiskit.result import Result

from ...capsule.mori import TagList


class After(NamedTuple):
    """`dataStateDepending` and `dataNeccessary` in V4 format."""

    retrievedResult: TagList[Hashable, Result]
    """The list of retrieved results, which multiple experiments shared."""
    allCounts: dict[str, list[dict[str, int]]]
    """The dict of all counts of each experiments.

    This attribute has been deprecated since version 0.13.0.
    Please find counts from the afterwards in :cls:`ExperimentPrototype` instead.
    """

    @staticmethod
    def _exporting_name() -> dict[str, str]:
        """The exporting name of :cls:`After`."""
        return {
            "retrievedResult": "retrievedResult",
            "allCounts": "allCounts",
        }
