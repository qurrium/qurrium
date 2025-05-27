"""Experiment - Afterwards (:mod:`qurry.qurrium.experiment.afterwards`)"""

import gc
import json
from typing import NamedTuple, Any, Union
from pathlib import Path
import warnings

from qiskit.result import Result

from ...capsule import DEFAULT_ENCODING
from ...exceptions import QurryResetSecurityActivated, QurryResetAccomplished


class After(NamedTuple):
    """The data of experiment will be independently exported in the folder 'legacy',
    which generated after the experiment."""

    # Measurement Result
    result: list[Result]
    """Results of experiment."""
    counts: list[dict[str, int]]
    """Counts of experiment."""

    @staticmethod
    def default_value():
        """The default value of each field."""
        return {
            "result": [],
            "counts": [],
        }

    @classmethod
    def read(
        cls,
        file_index: dict[str, str],
        save_location: Path,
    ) -> "After":
        """Read the exported experiment file.

        Args:
            file_index (dict[str, str]): The index of exported experiment file.
            save_location (Path): The location of exported experiment file.

        Returns:
            tuple[dict[str, Any], "After", dict[str, Any]]:
                The experiment's arguments,
                the experiment's common parameters,
                and the experiment's side product.
        """
        raw_data = {}
        with open(save_location / file_index["legacy"], encoding=DEFAULT_ENCODING) as f:
            raw_data = json.load(f)
        legacy: dict[str, Any] = raw_data["legacy"]
        for k, dv in cls.default_value().items():
            if k not in legacy:
                legacy[k] = dv

        return cls(**legacy)

    def export(self) -> dict[str, Any]:
        """Export the experiment's data after executing.

        Returns:
            dict[str, Any]: The experiment's data after executing.
        """
        return {"counts": self.counts}

    def clear_result(self, *args, security: bool = False, mute_warning: bool = True):
        """Clear the result of experiment.

        Args:
            security (bool, optional): Security for clearing. Defaults to `False`.
            mute_warning (bool, optional): Mute the warning when clearing. Defaults to `False`.
        """
        if len(args) > 0:
            warnings.warn(
                "'clear_result' is called, "
                + "does not execute to prevent executing accidentally."
                + "If you are sure to clear the result, use '.(security=True)'."
                + "Also, position arguments is not supported.",
                QurryResetSecurityActivated,
            )

        if security and isinstance(security, bool):
            self.result.clear()
            if not mute_warning:
                warnings.warn(
                    "The result of experiment is cleared.",
                    QurryResetAccomplished,
                )
            gc.collect()
        else:
            warnings.warn(
                "'clear_result' is called, but does not execute to prevent executing accidentally."
                + "If you are sure to clear the result, please set .(security=True).",
                QurryResetSecurityActivated,
            )


def create_afterwards(
    after: Union[After, None] = None,
) -> After:
    """Create an Afterwards object.

    Args:
        after (Union[After, None], optional):
            The After object to create. Defaults to None.

    Returns:
        After: The After object.
    """
    if after is None:
        return After(**After.default_value())
    if isinstance(after, After):
        return after

    raise TypeError("The 'after' must be an instance of 'After' or None.")
