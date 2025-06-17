"""TagList (:mod:`qurry.capsule.mori.taglist`)"""

from typing import Optional, Union, NamedTuple, Any, TypeVar, overload
from collections import defaultdict
from collections.abc import Hashable, Iterable
from pathlib import Path
import os
import json
import warnings

from ..utils import OpenArgs, JSONDumpArgs, create_open_args, create_json_dump_args
from ..jsonablize import parse
from ..exception import TagListTakeNotIterableWarning


_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")


def tuple_str_parse(k: str) -> Union[tuple[str, ...], str]:
    """Convert tuple strings to real tuple.

    Args:
        k (str): Tuplizing available string.

    Returns:
        Union[tuple[str, ...], str]: Result of tuplizing.
    """
    if k[0] == "(" and k[-1] == ")":
        kt = list(k[1:-1].split(","))
        kt2 = []
        for ktsub in kt:
            if len(ktsub) > 0:
                if ktsub[0] == "'" or ktsub[0] == '"':
                    kt2.append(ktsub[1:-1].strip())
                elif ktsub.isdigit():
                    kt2.append(int(ktsub))
                else:
                    kt2.append(ktsub)

        kt2 = tuple(kt2)
        return kt2
    return k


@overload
def key_tuple_loads(o: _T) -> _T: ...
@overload
def key_tuple_loads(o: dict[_K, _T]) -> dict[_K, _T]: ...
@overload
def key_tuple_loads(o: dict[Hashable, _T]) -> dict[Hashable, _T]: ...


def key_tuple_loads(o):
    """If a dictionary with string keys
    which read from json may originally be a python tuple,
    then transplies as a tuple.

    Args:
        o (dict): A dictionary with string keys which read from json.

    Returns:
        dict: Result which turns every possible string keys returning to 'tuple'.
    """

    if not isinstance(o, dict):
        return o

    ks = list(o.keys())
    for k in ks:
        if isinstance(k, str):
            kt2 = tuple_str_parse(k)
            if kt2 != k:
                o[kt2] = o[k]
                del o[k]
    return o


class TagList(defaultdict[_K, Union[list[_V], list[Any]]]):
    """Specific data structures of :mod:`qurrium` like `dict[str, list[any]]`.

    >>> bla = TagList()

    >>> bla.guider('strTag1', [...])
    >>> bla.guider(('tupleTag1', ), [...])
    >>> # other adding of key and value via `.guider()`
    >>> bla
    ... {
    ...     (): [...], # something which does not specify tags.
    ...     'strTag1': [...], # something
    ...     ('tupleTag1', ): [...],
    ...     ... # other hashable as key in python
    ... }

    Args:
        name (str, optional):
            The name of this :cls:`TagList`. Defaults to `TagList`.

    Raises:
        ValueError: When input is not a dict.
    """

    __name__ = "TagList"
    protect_keys = ["_all", ()]

    def __init__(
        self,
        o: Optional[dict[_K, Iterable[Any]]] = None,
        name: str = __name__,
        tuple_str_auto_transplie: bool = True,
    ) -> None:
        pass_o = {} if o is None else o
        if not isinstance(pass_o, dict):
            raise ValueError("Input needs to be a dict with all values are iterable.")
        super().__init__(list)
        self.__name__ = name

        pass_o = key_tuple_loads(pass_o) if tuple_str_auto_transplie else pass_o
        not_list_v = []
        for k, v in pass_o.items():
            if isinstance(v, Iterable):
                self[k].extend(v)  # type: ignore
            else:
                not_list_v.append(k)

        if len(not_list_v) > 0:
            warnings.warn(
                f"The following keys '{not_list_v}' "
                + "with the values are not iterable won't be added.",
                category=TagListTakeNotIterableWarning,
            )

    def all(self) -> list[_V]:
        """Export all values in `tagList`.

        Returns:
            list: All values in `tagList`.
        """
        d = []
        for v in self.values():
            if isinstance(v, list):
                d += v
        return d

    def guider(self, proposal_tag: Optional[_K] = None, v: Any = None) -> None:
        """Append a value to the :cls:`TagList` with a tag.

        Args:
            proposal_tag (any): The tag for this value.
            v (any): The value for legacy.
        """
        for k in self.protect_keys:
            if proposal_tag == k:
                raise ValueError(f"'{k}' is a reserved key for export data.")

        if proposal_tag is None:
            self[()].append(v)  # type: ignore
        elif proposal_tag in self:
            self[proposal_tag].append(v)
        else:
            self[proposal_tag] = [v]

    class ParamsControl(NamedTuple):
        """The type of arguments for :func:`params_control`"""

        open_args: OpenArgs
        """ The arguments for :func:`open` function.
        Defaults to DEFAULT_OPEN_ARGS, which is:
        >>> {
            'mode': 'w+',
            'encoding': 'utf-8',
        }
        """
        json_dump_args: JSONDumpArgs
        """The arguments for :func:`json.dump` function.
        Defaults to DEFAULT_JSON_DUMP_ARGS, which is:
        >>> {
            'indent': 2,
        }
        """
        save_location: Path
        """The exported location. Defaults to `Path('./')`."""

    @classmethod
    def params_control(
        cls,
        open_args: Optional[OpenArgs] = None,
        json_dump_args: Optional[JSONDumpArgs] = None,
        save_location: Union[Path, str] = Path("./"),
        is_read_only: bool = False,
    ) -> ParamsControl:
        """Handling all arguments.

        Args:
            open_args (Optional[OpenArgs], optional):
                The other arguments for :func:`open` function.
                Defaults to DEFAULT_OPEN_ARGS, which is:
                >>> {
                    'mode': 'w+',
                    'encoding': 'utf-8',
                }
            json_dump_args (Optional[JSONDumpArgs], optional):
                The other arguments for :func:`json.dump` function.
                Defaults to DEFAULT_JSON_DUMP_ARGS, which is:
                >>> {
                    'indent': 2,
                }
            save_location (Path, optional):
                The exported location. Defaults to `Path('./')`.
            is_read_only (bool, optional):
                Whether to read a file of :cls:`TagList` exportation.

        Returns:
            ParamsControl: Current arguments.
        """

        open_args = create_open_args(open_args=open_args, is_read_only=is_read_only)
        json_dump_args = create_json_dump_args(json_dump_args=json_dump_args)

        if isinstance(save_location, str):
            save_location = Path(save_location)
        elif isinstance(save_location, Path):
            ...
        else:
            raise ValueError("'save_location' needs to be the type of 'str' or 'Path'.")

        if not os.path.exists(save_location):
            raise FileNotFoundError(f"Such location not found: {save_location}")

        return cls.ParamsControl(
            open_args=open_args,
            json_dump_args=json_dump_args,
            save_location=save_location,
        )

    def export(
        self,
        name: Optional[str],
        save_location: Union[Path, str] = Path("./"),
        taglist_name: str = __name__,
        open_args: Optional[OpenArgs] = None,
        json_dump_args: Optional[JSONDumpArgs] = None,
    ) -> Path:
        """Export :cls:`TagList` to a file.

        Args:
            name (Optional[str], optional):
                The file name should be something like
                "{name}.{taglist_name}.json" or "{taglist_name}.json" when 'name' is None.
            save_location (Path):
                The location of file. Defaults to `Path('./')`.
            taglist_name (str, optional):
                The suffix name for this `tagList`.
                Defaults to `__name__`.
                The file name will be
                "{name}.{taglist_name}.json" or "{taglist_name}.json" when 'name' is None.
            open_args (Optional[OpenArgs], optional):
                The other arguments for :func:`open` function.
                Defaults to DEFAULT_OPEN_ARGS, which is:
                >>> {
                    'mode': 'w+',
                    'encoding': 'utf-8',
                }
            json_dump_args (Optional[JSONDumpArgs], optional):
                The other arguments for :func:`json.dump` function.
                Defaults to DEFAULT_JSON_DUMP_ARGS, which is:
                >>> {
                    'indent': 2,
                }

        Raises:
            ValueError: When filetype is not supported.

        Return:
            Path: The path of exported file.
        """

        args = self.params_control(
            open_args=open_args,
            json_dump_args=json_dump_args,
            save_location=save_location,
        )
        assert "encoding" in args.open_args, "encoding must be specified in open_args"
        encoding = args.open_args.pop("encoding")
        filename = f"{taglist_name}.json" if name is None else f"{name}.{taglist_name}.json"
        assert "encoding" not in args.open_args, "encoding must not be in open_args after pop"

        with open(
            args.save_location / filename, encoding=encoding, **args.open_args
        ) as export_json:
            json.dump(parse(self), export_json, **args.json_dump_args)

        return args.save_location / filename

    @classmethod
    def read(
        cls,
        filename: str,
        save_location: Union[Path, str] = Path("./"),
        taglist_name: str = __name__,
        tuple_str_auto_transplie: bool = True,
        open_args: Optional[OpenArgs] = None,
        json_dump_args: Optional[JSONDumpArgs] = None,
    ) -> "TagList":
        """Read a :cls:`TagList` from a file.

        Args:
            filename (str):
                The file name of exported :cls:`TagList`.
                The file name should be something like
                "{name}.{taglist_name}.json" or "{taglist_name}.json" when 'name' is None.
            save_location (Path):
                The location of file. Defaults to `Path('./')`.
            taglist_name (str, optional):
                The class name of :cls:`TagList`.
                Defaults to `__name__`.
            tuple_str_auto_transplie (bool, optional):
                Whether to transplie tuple string in the keys of the dict
                to real tuple. Defaults to True.
            open_args (Optional[OpenArgs], optional):
                The other arguments for :func:`open` function.
                Defaults to DEFAULT_OPEN_ARGS, which is:
                >>> {
                    'mode': 'w+',
                    'encoding': 'utf-8',
                }
            json_dump_args (Optional[JSONDumpArgs], optional):
                The other arguments for :func:`json.dump` function.
                Defaults to DEFAULT_JSON_DUMP_ARGS, which is:
                >>> {
                    'indent': 2,
                }

        Raises:
            FileNotFoundError: When file not found.

        Return:
            TagList: The path of exported file.
        """

        args = cls.params_control(
            open_args=open_args,
            json_dump_args=json_dump_args,
            save_location=save_location,
            is_read_only=True,
        )
        assert "encoding" in args.open_args, "encoding must be specified in open_args"
        encoding = args.open_args.pop("encoding")
        assert "encoding" not in args.open_args, "encoding must not be in open_args after pop"

        with open(args.save_location / filename, encoding=encoding, **args.open_args) as read_json:
            raw_data = json.load(read_json)
            obj = cls(
                o=raw_data,
                name=taglist_name,
                tuple_str_auto_transplie=tuple_str_auto_transplie,
            )
        return obj
