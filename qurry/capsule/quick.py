"""Quick CapSule (:mod:`qurry.capsule.quick`)"""

from typing import TypeVar, Union, Optional, Literal, Any, overload
from pathlib import Path
from json import JSONDecoder, load

# pylint: disable=invalid-name

JsonDecoderType = TypeVar("JsonDecoderType", bound=JSONDecoder)


@overload
def quickRead(filename: Union[str, Path]) -> Any: ...


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]] = None,
    *,
    filetype: Literal["json"] = "json",
    encoding: str = "utf-8",
    cls: None = None,
) -> Any: ...


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]] = None,
    *,
    filetype: Literal["txt"],
    encoding: str = "utf-8",
    cls: Optional[JsonDecoderType] = None,
) -> str: ...


@overload
def quickRead(
    filename: Union[str, Path],
    save_location: Optional[Union[str, Path]] = None,
    *,
    filetype: Literal["json"] = "json",
    encoding: str = "utf-8",
    cls: JsonDecoderType,
) -> JsonDecoderType: ...


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
                return load(File, cls=cls)
        with open(save_location / filename, "r", encoding=encoding) as File:
            return load(File)

    else:
        with open(save_location / filename, "r", encoding=encoding) as File:
            return File.read()


# pylint: enable=invalid-name
