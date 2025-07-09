"""Quick CapSule (:mod:`qurry.capsule.quick`)"""

from typing import Literal, Any, TypeVar, Optional, Type, overload, Union
from pathlib import Path
import json

# pylint: disable=invalid-name

JsonDecoderType = TypeVar("JsonDecoderType", bound=Type[json.JSONDecoder])


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]],
    filetype: Literal["json"],
    encoding: str,
    cls: JsonDecoderType,
) -> JsonDecoderType: ...


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]],
    filetype: Literal["json"],
    encoding: str,
    cls: None,
) -> Any: ...


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]],
    filetype: Literal["txt"],
    encoding: str,
    cls: Optional[JsonDecoderType],
) -> str: ...


def quickRead(
    filename,
    save_location=None,
    filetype="json",
    encoding="utf-8",
    cls=None,
):
    """Quick read file.

    Args:
        filename (Union[str, Path]):
            Filename.
        save_location (Optional[Union[str, Path]], optional):
            Location to save the file. Defaults to Path("./").
        filetype (Literal["json", "txt"], optional):
            Type of the file. Defaults to "json".
        encoding (str, optional):
            Encoding method. Defaults to 'utf-8'.
        cls (Optional[JsonDecoderType], optional):
            Custom JSON decoder class. Defaults to None.

    Returns:
        Content of the file.
    """
    if save_location is None:
        save_location = Path("./")
    if not isinstance(save_location, Path):
        save_location = Path(save_location)

    if filetype == "json":
        if cls is not None:
            with open(save_location / filename, "r", encoding=encoding) as File:
                return json.load(File, cls=cls)
        with open(save_location / filename, "r", encoding=encoding) as File:
            return json.load(File)

    else:
        with open(save_location / filename, "r", encoding=encoding) as File:
            return File.read()


# pylint: enable=invalid-name
