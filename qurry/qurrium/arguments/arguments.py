"""The Arguments of Experiment (:mod:`qurry.qurrium.experiment.arguments`)"""

from typing import Union, Any, TypeVar
from collections.abc import Iterable
from dataclasses import dataclass, fields

from .utils import filter_deprecated_args, create_exp_outfields
from .commonparams import Commonparams


@dataclass(frozen=True)
class ArgumentsPrototype:
    """Construct the experiment's parameters for specific options,
    which is overwritable by the inherition class."""

    exp_name: str
    """Name of experiment."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(self.__dict__.keys())

    def _asdict(self) -> dict[str, Any]:
        """The arguments as dictionary."""
        return self.__dict__

    @classmethod
    def _dataclass_fields(cls) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def _make(cls, iterable: Iterable):
        """Make the arguments."""
        return cls(*iterable)

    @classmethod
    def _filter(cls, *args, **kwargs):
        """Filter the arguments of the experiment.

        Args:
            *args: The arguments of the experiment.
            **kwargs: The keyword arguments of the experiment.

        Returns:
            tuple[ArgumentsPrototype, Commonparams, dict[str, Any]]:
                The arguments of the experiment,
                the common parameters of the experiment,
                and the side product of the experiment.
        """
        if len(args) > 0:
            raise ValueError("args filter can't be initialized with positional arguments.")
        infields = {}
        commonsinput = {}
        outfields = {}
        for k, v in kwargs.items():
            # pylint: disable=protected-access
            if k in cls._dataclass_fields():
                # pylint: enable=protected-access
                infields[k] = v
            elif k in Commonparams._fields:
                commonsinput[k] = v
            else:
                outfields[k] = v

        return (cls(**infields), Commonparams(**commonsinput), outfields)  # type: ignore

    @classmethod
    def create(cls, arguments: Union["_A", dict[str, Any]]):
        """Create experiment arguments from the given arguments.

        Args:
            arguments (Union[_A, dict[str, Any]]): The arguments to be parsed.
            arguments_instance (type[_A]): The instance of the arguments class.

        Raises:
            TypeError: If the arguments is not an instance of the arguments class or a dictionary.

        Returns:
            A tuple containing the parsed arguments instance and a dictionary of deprecated fields.
        """

        if isinstance(arguments, cls):
            return arguments, {}
        if isinstance(arguments, dict):
            # pylint: disable=protected-access
            arg_parsed, arguments_deprecated = filter_deprecated_args(
                arguments, cls._dataclass_fields()
            )
            # pylint: enable=protected-access
            return cls(**arg_parsed), arguments_deprecated

        raise TypeError(f"arguments should be {cls} or dict, not {type(arguments)}")


_A = TypeVar("_A", bound=ArgumentsPrototype)
"""Type variable for :class:`ArgumentsPrototype`."""


def create_all_arguments(
    arguments: Union[_A, dict[str, Any]],
    commonparams: Union[Commonparams, dict[str, Any]],
    outfields: dict[str, Any],
    arguments_instance: type[_A],
) -> tuple[_A, Commonparams, dict[str, Any]]:
    """Create experiment arguments from the given arguments.

    Args:
        arguments (Optional[Union[NamedTuple, dict[str, Any]]]):
            The arguments of the experiment.
        commonparams (Optional[Union[Commonparams, dict[str, Any]]]):
            The common parameters of the experiment.
        outfields (Optional[dict[str, Any]]):
            The outfields of the experiment.
        arguments_instance (type[_A]):
            The instance of the arguments class.

    Returns:
        The arguments of the experiment,
        the common parameters of the experiment,
        and the side product of the experiment.
    """

    the_arguments, arguments_deprecated = arguments_instance.create(arguments)
    commons, commonparams_deprecated = Commonparams.create(commonparams)
    outfields = create_exp_outfields(outfields)

    if len(arguments_deprecated):
        outfields["arguments_deprecated"] = arguments_deprecated
    if len(commonparams_deprecated):
        outfields["commonparams_deprecated"] = commonparams_deprecated

    return the_arguments, commons, outfields
