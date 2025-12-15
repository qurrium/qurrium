"""Tuple Key Parser (:mod:`qurry.capsule.mori.tuple_key`)"""

from typing import Union, TypeVar, overload
from collections.abc import Hashable


_K = TypeVar("_K", bound=Hashable)
_T = TypeVar("_T")


def tuple_str_parse(kstring: str) -> Union[tuple[str, ...], str]:
    r"""Convert tuple strings to real tuple.

    >>> tuple_str_parse("hello_world")
    'hello_world'

    >>> tuple_str_parse("(1, 2, 3)")
    (1, 2, 3)

    >>> tuple_str_parse("('a', 'b', 'c')")
    ('a', 'b', 'c')

    >>> tuple_str_parse("('delay', '1.60e+01_ns', 'excited')")
    ('delay', '1.60e+01_ns', 'excited')

    >>> tuple_str_parse(
        '(\'normalstr\', \'alsostr\', \'"power"str\', "p\'aw\'a", 42, \'114514\', \'\')'
    )
    ('normalstr', 'alsostr', '"power"str', "p'aw'a", 42, '114514', '')

    Args:
        kstring (str): Tuplizing available string.

    Returns:
        Union[tuple[str, ...], str]: Result of tuplizing.
    """
    if not isinstance(kstring, str):
        raise ValueError("Input must be a string")

    if kstring[0] != "(" or kstring[-1] != ")":
        return kstring

    kt = list(kstring[1:-1].split(", "))
    # ", " is the acutal divider of the tuple elements
    # Not just comma alone
    # Otherwise it will make some really bad result...
    kt2 = []
    for ktelt in kt:
        if len(ktelt) <= 0:
            continue
        if ktelt[0] == "'" or ktelt[0] == '"':
            kt2.append(ktelt[1:-1].strip())
        elif ktelt.isdigit():
            kt2.append(int(ktelt))
        else:
            kt2.append(ktelt)

    return tuple(kt2)


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
