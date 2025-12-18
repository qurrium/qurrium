"""The Side Product Container (:mod:`qurry.qurrium.experiment.tales`)"""

import json
from typing import Any, TypeVar, Generic, cast
from pathlib import Path
import warnings

from ..utils.file_structure import (
    FOLDER_NAME_SIDE_PRODUCTS as FOLDER_NAME,
    is_old_v7_file_structure,
)
from ..exceptions import OldFormatedIncompatibleWarning, MSG_V7_FILE_FORMAT_INCOMPATIBLE
from ...capsule import jsonablize, DEFAULT_ENCODING, CustomDict
from ...capsule.mori import FileReadableWritableObj, WrittenContentType


FILENAME_TEMPLATE = "{}.tales.json"
"""Filename template for side products export."""

_SPT = TypeVar("_SPT")
"""Type variable for side product types. This made for :class:`~typing.TypedDict`.
For example:

.. code-block:: python

    from typing import TypedDict

    class MySchema(TypedDict):
        count: int
        name: str

    tales: Tales[MySchema] = Tales({"count": 42, "name": "test"})
    typed_data = tales.as_typed()
    print(typed_data["count"])

>>> 42

"""


class Tales(CustomDict, FileReadableWritableObj, Generic[_SPT]):
    """A customized dictionary for storing side products.

    This supports type checking with :class:`~typing.TypedDict`.

    .. code-block:: python

        from typing import TypedDict

        class MySchema(TypedDict):
            count: int
            name: str

        tales: Tales[MySchema] = Tales({"count": 42, "name": "test"})
        typed_data = tales.as_typed()
        print(typed_data["count"])

    >>> 42
    """

    def as_typed(self) -> _SPT:
        """Return self as the typed version for type checking."""
        return cast(_SPT, self)

    @classmethod
    def remain_keys(cls) -> tuple[str, ...]:
        """The keys that will be remained in the side product container,
        which denotes with the typed dictionary.
        """
        return ()

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
        """Process the serialized content from the method :meth:`content_writing`
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
            if is_old_v7_file_structure(file_index):
                warnings.warn(MSG_V7_FILE_FORMAT_INCOMPATIBLE, OldFormatedIncompatibleWarning)
                return cls()
            raise KeyError(f"The '{FOLDER_NAME}' field is missing in the file index.")

        with open(save_location / file_index[FOLDER_NAME], "r", encoding=DEFAULT_ENCODING) as f:
            side_products = cls.content_loading(json.load(f))

        return cls(side_products)


_SP = TypeVar("_SP", bound="Tales")
"""Type variable for Tales class, the side product container."""
