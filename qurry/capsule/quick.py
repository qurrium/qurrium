"""Quick CapSule (:mod:`qurry.capsule.quick`)"""

from typing import Union, Literal, Any
from pathlib import Path
import json

# pylint: disable=invalid-name


def quickRead(
    filename: Union[str, Path],
    save_location: Union[Path, str] = Path("./"),
    filetype: Literal["json", "txt"] = "json",
    encoding: str = "utf-8",
) -> Any:
    """Quick read file.

    Args:
        filename (Union[str, Path]): Filename.
        encoding (str, optional): Encoding method. Defaults to 'utf-8'.

    Returns:
        str: Content of the file.
    """
    if not isinstance(save_location, Path):
        save_location = Path(save_location)

    if filetype == "json":
        with open(save_location / filename, "r", encoding=encoding) as File:
            return json.load(File)

    else:
        with open(save_location / filename, "r", encoding=encoding) as File:
            return File.read()


# pylint: enable=invalid-name
