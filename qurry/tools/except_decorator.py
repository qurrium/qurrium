"""Eception Decorator (:mod:`qurry.tools.except_decorator`)"""

import functools
import warnings
import inspect
from typing import Callable, Union, Type, TypeVar

from ..exceptions import QurryUnprovenFeatureWarning

U = TypeVar("U", bound=Union[Type, Callable])


def unproven_feature(message=None):
    """The decorator to mark a function or class as an unproven feature.

    Args:
        message (Optional[str]):
            The warning message to be displayed.
            If not provided, a default message will be used.
            The default message is:
            "This feature is unproven and may be unstable or behave inconsistently.
            Use with caution."

    Returns:
        Union[Callable, Type]:

    Examples:
        >>> @unproven_feature
        ... def my_experimental_function():
        ...     pass

        >>> @unproven_feature(message="This is a custom message.")
        ... class MyExperimentalClass:
        ...     pass
    """

    def decorator(func_or_cls: U) -> U:
        """The actual decorator function.
        Args:
            func_or_cls (Union[Callable, Type]):
                The function or class to be marked.
        Returns:
            Union[Callable, Type]:
        """

        name = func_or_cls.__qualname__

        warn_message = (
            message or f"This feature '{name}' is unproven, we can not guarantee the correctness."
        )

        if inspect.isclass(func_or_cls):
            original_init = func_or_cls.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                warnings.warn(warn_message, QurryUnprovenFeatureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            func_or_cls.__init__ = wrapped_init

            func_or_cls.__unproven_feature__ = True

            return func_or_cls  # type: ignore[return-value]

        if inspect.isfunction(func_or_cls):
            warnings.warn(warn_message, QurryUnprovenFeatureWarning, stacklevel=2)
            return func_or_cls  # type: ignore[return-value]

        raise TypeError(
            "The decorator can only be applied to functions or classes."
            + f" Got {type(func_or_cls)}."
        )

    return decorator
