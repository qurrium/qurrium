"""Utility functions and type definitions for Mori capsule.
(:mod:`qurry.capsule.mori.utils`)
"""

from typing import TypedDict, Union, Callable, Any, Optional
import warnings
from json import JSONEncoder

DEFAULT_ENCODING = "utf-8"
DEFAULT_INDENT = 2
DEFAULT_MODE = "w+"
DEFAULT_ENSURE_ASCII = False


class OpenArgs(TypedDict, total=False):
    """Default arguments for open function."""

    mode: str
    buffering: int
    encoding: Optional[str]
    errors: Optional[str]
    newline: Optional[str]
    closefd: bool
    opener: Optional[Callable[[str, int], int]]


DEFAULT_OPEN_ARGS: OpenArgs = {
    "mode": DEFAULT_MODE,
    "encoding": DEFAULT_ENCODING,
}


def create_open_args(
    open_args: Union[dict[str, Any], OpenArgs, None] = None,
    is_read_only: bool = False,
) -> OpenArgs:
    """Create open arguments.
    Args:
        open_args (Union[dict[str, Any], OpenArgs]): Arguments for open function.
        is_read_only (bool, optional): Whether the file is read-only. Defaults to False.
    Raises:
        TypeError: If 'open_args' is not a dict.
    Returns:
        OpenArgs: The open arguments.
    """

    new_open_args = DEFAULT_OPEN_ARGS.copy()
    if open_args is not None:
        if not isinstance(open_args, dict):
            raise TypeError("'open_args' must be a dict.")
        if open_args.pop("file", None):
            warnings.warn(
                "Argument 'file' is ignored for it will be used by 'TagList.export'.",
                UserWarning,
            )
        for k, v in open_args.items():
            new_open_args[k] = v
    open_args = new_open_args
    if is_read_only:
        open_args["mode"] = "r"

    return open_args


class PrintArgs(TypedDict, total=False):
    """Default arguments for print function."""

    flush: bool
    end: str
    sep: str


DEFAULT_PRINT_ARGS: PrintArgs = {}


def create_print_args(
    print_args: Union[dict[str, Any], PrintArgs, None] = None,
) -> PrintArgs:
    """Create print arguments.

    Args:
        print_args (Union[dict[str, Any], PrintArgs]): Arguments for print function.

    Returns:
        PrintArgs: The print arguments.
    """

    new_print_args = DEFAULT_PRINT_ARGS.copy()
    if print_args is not None:
        if not isinstance(print_args, dict):
            raise TypeError("'print_args' must be a dict.")
        for k, v in print_args.items():
            if k == "file":
                warnings.warn(
                    "Argument 'file' is ignored for it will be used by 'TagList.export'.",
                    UserWarning,
                )
            new_print_args[k] = v
    return new_print_args


class JSONDumpArgs(TypedDict, total=False):
    """Default arguments for print function."""

    skipkeys: bool
    ensure_ascii: bool
    check_circular: bool
    allow_nan: bool
    cls: Optional[type[JSONEncoder]]
    indent: Union[int, str, None]
    separators: Optional[tuple[str, str]]
    default: Optional[Callable[[Any], Any]]
    sort_keys: bool


DEFAULT_JSON_DUMP_ARGS: JSONDumpArgs = {
    "indent": DEFAULT_INDENT,
    "ensure_ascii": DEFAULT_ENSURE_ASCII,
}


def create_json_dump_args(
    json_dump_args: Union[dict[str, Any], JSONDumpArgs, None] = None,
) -> JSONDumpArgs:
    """Create JSON dump arguments.

    Args:
        json_dump_args (Union[dict[str, Any], JSONDumpArgs]): Arguments for json.dump function.

    Returns:
        JSONDumpArgs: The JSON dump arguments.
    """

    new_json_dump_args = DEFAULT_JSON_DUMP_ARGS.copy()
    if json_dump_args is not None:
        if not isinstance(json_dump_args, dict):
            raise TypeError("'json_dump_args' must be a dict.")
        for k, v in json_dump_args.items():
            if k in ["obj", "fp"]:
                warnings.warn(
                    f"Argument '{k}' is ignored for it will be used by 'TagList.export'.",
                    UserWarning,
                )
            new_json_dump_args[k] = v
    return new_json_dump_args
