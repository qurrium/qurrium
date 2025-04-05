"""
================================================================
Eception Decorator (:mod:`qurry.tools.except_decorator`)
================================================================
"""

import functools
import warnings
from typing import Any, Callable, Type, TypeVar, overload

from ..exceptions import QurryUnprovenFeatureWarning

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=Type[Any])


@overload
def unproven_feature(func: F) -> F: ...
@overload
def unproven_feature(*, message: str = None) -> Callable[[F], F]: ...
@overload
def unproven_feature(cls: C) -> C: ...


def unproven_feature(func_or_cls=None, *, message=None):
    """The decorator to mark a function or class as an unproven feature.

    Args:
        func_or_cls (Optional[Union[Callable, Type]]):
            The function or class to be marked.
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

    def decorator(func_or_cls):
        # 獲取函數或類別名稱
        name = func_or_cls.__qualname__

        # 確定警告訊息
        warn_message = (
            message
            or f"This feature '{name}' is unproven and may be unstable or behave inconsistently."
            + "Use with caution."
        )

        if isinstance(func_or_cls, type):
            original_init = func_or_cls.__init__

            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                warnings.warn(warn_message, QurryUnprovenFeatureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            func_or_cls.__init__ = wrapped_init

            func_or_cls.__unproven_feature__ = True

            return func_or_cls

        @functools.wraps(func_or_cls)
        def wrapper(*args, **kwargs):
            warnings.warn(warn_message, QurryUnprovenFeatureWarning, stacklevel=2)
            return func_or_cls(*args, **kwargs)

        wrapper.__unproven_feature__ = True

        return wrapper

    if func_or_cls is None:
        return decorator
    return decorator(func_or_cls)
