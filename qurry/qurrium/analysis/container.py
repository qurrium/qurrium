"""AnalysisContainer (:mod:`qurry.qurrium.experiment.analyses`)"""

from typing import Any, Optional
from pathlib import Path
import json

from .analysis import _R
from ..json_io import FileReadableWritableObj, WrittenContentType
from ...capsule import DEFAULT_ENCODING


FOLDER_NAME = "myths"
"""Folder name for analyses export."""
FILENAME_TEMPLATE = "{}.myths.json"
"""Filename template for analyses export."""


class AnalysesContainer(dict[int, _R], FileReadableWritableObj):
    """A customized dictionary for storing
    :class:`~qurry.qurrium.analysis.AnalysisPrototype` objects."""

    __name__ = "AnalysesContainer"

    def __init__(self, *, analysis_instance: type[_R]):
        self.analysis_instance = analysis_instance
        super().__init__()

    def export(self):
        """Export the analyses for file writing.

        Returns:
            dict[int, dict[str, Any]]: The exported analyses.
        """

        return {k: v.export() for k, v in self.items()}

    @classmethod
    def folder_and_filename(cls, identifier: str) -> tuple[str, str]:
        """Get the folder name and filename for the given analysis ID.

        Args:
            identifier (str): Identifier for the experiments.

        Returns:
            tuple[str, str]: The folder name and filename for the experiments.
        """
        return FOLDER_NAME, FILENAME_TEMPLATE.format(identifier)

    def content_writing(self) -> WrittenContentType[dict[int, dict[str, Any]]]:
        """Get the content to be written to files.

        Returns:
            WritingContentType: The content to be written to files.
        """
        return {"reports": self.export()}

    @classmethod
    def load(cls, raw_dict: dict[str, Any]):
        """Load from a raw dictionary.
        Also works for the inner process in object_hook in json.load.

        Args:
            raw_dict (dict[str, Any]): The dictionary to deserialize.

        Returns:
            The deserialized analysis instance, or None if not applicable.
        """

        return {int(k): cls.load(v) for k, v in raw_dict.items()}

    @classmethod
    def content_loading(cls, raw_read: dict[str, Any]):
        """The object hook for json.load.

        Args:
            raw_read (dict[str, Any]): The raw read dictionary.

        Returns:
            The deserialized analysis instance, or None if not applicable.
        """
        if "report" not in raw_read:
            raise KeyError("The 'report' field is missing in the raw read data.")

        return cls.load(raw_read["report"])

    @classmethod
    def read(cls, file_index: dict[str, str], save_location: Path):
        """Read the analysis from file index.

        Args:
            file_index (dict[str, str]): The file index.
            save_location (Path): The save location.

        Returns:
            The analysis instances in dictionary.
        """
        if "reports" not in file_index:
            raise KeyError("The file index does not contain 'reports' key.")

        with open(save_location / file_index["reports"], "r", encoding=DEFAULT_ENCODING) as f:
            analyses_data: dict[int, _R] = json.load(f, object_hook=cls.content_loading)

        return analyses_data

    def __repr__(self):
        inner_lines = ", ".join(f"{k}" + "{...}" for k in self.keys())
        return f"{self.__name__}(length={len(self)}, " + "{" + f"{inner_lines}" + "})"

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__name__}(length={len(self)}, ...)")
        else:
            with p.group(2, f"{self.__name__}(length={len(self)}" + ", {"):
                p.breakable()
                for i, (k, v) in enumerate(self.items()):
                    if i:
                        p.text(",")
                        p.breakable()
                    p.text(f"{k}: {v}")
                p.text("})")

    @classmethod
    def create(
        cls, reports: Optional["AnalysesContainer[_R]"], *, analysis_instance: type[_R]
    ) -> "AnalysesContainer[_R]":
        """Create an AnalysesContainer from the given reports.

        Args:
            reports (Optional[AnalysesContainer[_R]]): The reports to be parsed.
            analysis_instance (type[_R]): The analysis instance type.
        Returns:
            AnalysesContainer[_R]: The created AnalysesContainer.
        """
        if reports is None:
            return cls(analysis_instance=analysis_instance)

        if reports.analysis_instance is not analysis_instance:
            raise ValueError("The analysis instance type does not match.")

        return reports
