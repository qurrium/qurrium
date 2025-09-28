"""Post Processing - Classical Shadow - Utilities - Method Enum
(:mod:`qurry.process.classical_shadow.utils.method_enum`)

The abstract base class for method enums with utility functions.
"""

from typing import TypeVar, Type, List
from abc import abstractmethod, ABCMeta
from enum import EnumMeta, Enum

T = TypeVar("T", bound="BaseMethodEnum")


class EnumABCMeta(EnumMeta, ABCMeta):
    """A metaclass that combines EnumMeta and ABCMeta."""


class BaseMethodEnum(Enum, metaclass=EnumABCMeta):
    """Base class for method enums with utility functions."""

    @classmethod
    def get_all_methods(cls: Type[T]) -> List[str]:
        """Get a list of all available methods.

        Returns:
            list[str]: A list of method names.
        """
        return [method.value for method in cls]  # type: ignore[attr-defined]

    @classmethod
    def unknown_method_error_msg(cls: Type[T]) -> str:
        """Generate a ValueError for an unknown method.

        Returns:
            str: The error message.
        """
        return f"Unknown method. Supported methods are: {', '.join(cls.get_all_methods())}"

    @classmethod
    def value_error(cls) -> ValueError:
        """Generate a ValueError for an unknown method.

        Returns:
            ValueError: The ValueError with the error message.
        """
        return ValueError(cls.unknown_method_error_msg())

    @classmethod
    def from_string(cls: Type[T], method_str: str) -> T:
        """Convert a string to a enum member.

        Args:
            method_str (str): The string representation of the method.

        Returns:
            The corresponding enum member.

        Raises:
            ValueError: If the string does not correspond to any enum member.
        """

        for method in cls:  # type: ignore[attr-defined]
            if method.value == method_str:
                return method
        raise cls.value_error()

    @classmethod
    @abstractmethod
    def get_default(cls: Type[T]) -> T:
        """Get the default method.

        Returns:
            The default method.
        """
        raise NotImplementedError(f"{cls.__name__} must implement get_default()")
