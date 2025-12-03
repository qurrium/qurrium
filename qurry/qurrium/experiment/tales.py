"""The Side Product Container (:mod:`qurry.qurrium.experiment.tales`)"""

import json
from typing import Any, TypeVar
from pathlib import Path

from ...capsule import jsonablize
from ..json_io import FileReadableWritableObj, WrittenContentType
from ...capsule import DEFAULT_ENCODING


FOLDER_NAME = "tales"
"""Folder name for side products export."""
FILENAME_TEMPLATE = "{}.tales.json"
"""Filename template for side products export."""


class Tales(dict[str, Any], FileReadableWritableObj):
    """A customized dictionary for storing side products."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export(self) -> dict[str, Any]:
        """Export the side products for file writing.

        Returns:
            dict[str, Any]: The side products as a dictionary.
        """

        return jsonablize(self)

    @classmethod
    def folder_and_filename(cls, identifier: str) -> tuple[str, str]:
        """Get the folder name and filename for the given analysis ID.

        Args:
            identifier (str): Identifier for the experiments.

        Returns:
            tuple[str, str]: The folder name and filename for the experiments.
        """
        return FOLDER_NAME, FILENAME_TEMPLATE.format(identifier)

    def content_dumping(self) -> WrittenContentType[dict[str, Any]]:
        """Get the content to be written to files.

        Returns:
            WritingContentType: The content to be written to files.
        """
        return {"side_products": self.export()}

    @classmethod
    def ingest(cls, raw_dict: dict[str, Any]):
        """Ingest from a serialized dictionary.

        Args:
            raw_dict (dict[str, Any]): The raw read dictionary.
        """
        return cls(raw_dict.items())

    @classmethod
    def content_loading(cls, raw_read: dict[str, Any]):
        """The object hook for :func:`~json.load`.
        Handle the raw read dictionary with specific structure,
        which is same with the one used in :meth:`FileWritableObj.content_writing`.

        Args:
            raw_read (dict[str, Any]): The raw read dictionary.

        Returns:
            Tales: The side product container.
        """
        if "side_products" not in raw_read:
            raise KeyError("The 'side_products' field is missing in the raw read data.")

        side_products_dict: dict[str, Any] = raw_read["side_products"]
        return cls.ingest(side_products_dict)

    @classmethod
    def read(cls, file_index: dict[str, str], save_location: Path) -> "Tales":
        """Read the exported experiment file.

        Args:
            file_index (dict[str, str]): The index of exported experiment file.
            save_location (Path): The location of exported experiment file.

        Returns:
            Tales: The side product container.
        """
        if FOLDER_NAME not in file_index:
            raise KeyError(f"The '{FOLDER_NAME}' field is missing in the file index.")

        with open(save_location / file_index[FOLDER_NAME], "r", encoding=DEFAULT_ENCODING) as f:
            side_products = json.load(f, object_hook=cls.content_loading)

        return side_products


_SP = TypeVar("_SP", bound="Tales")
"""Type variable for Tales class, the side product container."""
