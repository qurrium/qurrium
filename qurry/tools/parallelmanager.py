"""The parallel tools for Qurry. (:mod:`qurry.tools.parallelmanager`)"""

import warnings
from typing import Optional, Iterable, Callable, TypeVar, Any, Literal
from multiprocessing import Pool, cpu_count, get_context
from tqdm.contrib.concurrent import process_map

from .progressbar import default_setup
from ..exceptions import QurryWarning


CPU_COUNT_UNSAFE = cpu_count()
"""The number of CPUs available for multiprocessing.
But this number may be None in some cases.
"""

CPU_COUNT = CPU_COUNT_UNSAFE if CPU_COUNT_UNSAFE else 1
"""The number of CPUs available for multiprocessing.
This number is guaranteed to be at least 1.
"""

DEFAULT_POOL_SIZE = CPU_COUNT
"""The default number of workers for multiprocessing.
This number is guaranteed to be at least 1.
"""


def workers_distribution(
    workers_num: Optional[int] = None,
    default: int = DEFAULT_POOL_SIZE,
) -> int:
    """Distribute the workers number.

    Args:
        workers_num (Optional[int], optional): Desired workers number. Defaults to None.
        default (int, optional): Default workers number. Defaults to DEFAULT_POOL_SIZE.

    Returns:
        int: Workers number.
    """

    if default < 1:
        warnings.warn(
            f"| Available worker number {CPU_COUNT} is equal orsmaller than 2."
            + "This computer may not be able to run this program for "
            + "the program will allocate all available threads.",
            category=QurryWarning,
        )
        default = DEFAULT_POOL_SIZE

    if workers_num is None:
        launch_worker = default
    else:
        if workers_num > CPU_COUNT:
            warnings.warn(
                f"| Worker number {workers_num} is larger than cpu count {CPU_COUNT}.",
                category=QurryWarning,
            )
            launch_worker = default
        elif workers_num < 1:
            warnings.warn(
                f"| Worker number {workers_num} is smaller than 1. Use single worker.",
                category=QurryWarning,
            )
            launch_worker = 1
        else:
            launch_worker = workers_num

    return launch_worker


# pylint: disable=invalid-name
T_map = TypeVar("T_map")
T_tgt = TypeVar("T_tgt")
# pylint: enable=invalid-name


class ParallelManager:
    """Process manager for multiprocessing."""

    def __init__(
        self,
        workers_num: Optional[int] = DEFAULT_POOL_SIZE,
        bar_format: str = "qurry-full",
        bar_ascii: str = "4squares",
        **pool_kwargs,
    ):
        """Initialize the process manager.

        Args:
            workers_num (Optional[int], optional):
                Desired workers number. Defaults to DEFAULT_POOL_SIZE.
            **pool_kwargs: Other arguments for Pool.
        """

        if "processes" in pool_kwargs:
            warnings.warn(
                "| `processes` is given in `pool_kwargs`."
                + "It will be overwritten by `workers_num`."
            )
            pool_kwargs.pop("processes")

        self.reslt_setup = default_setup(bar_format, bar_ascii)
        self.pool_kwargs = pool_kwargs
        self.workers_num = workers_distribution(workers_num)

    def starmap(
        self,
        func: Callable[..., T_map],
        args_list: Iterable,
        start_method: Optional[Literal["spawn", "fork", "forkserver"]] = None,
    ) -> list[T_map]:
        """This function is a wrapper for starmap from multiprocessing.

        Args:
            func (Callable[[Iterable[T_tgt]], T_map]): Function to be mapped.
            args_list (Iterable[Iterable[T_tgt]]): Arguments to be mapped.
            start_method (Optional[Literal["spawn", "fork", "forkserver"]], optional):
                Start method for multiprocessing. Defaults to None.
                If None, use the default start method of the system.

        Returns:
            tqdm.tqdm[T_map]: Results.
        """

        if self.workers_num == 1:
            return list(map(func, *zip(*args_list)))
        pool_instance = get_context(start_method).Pool if start_method else Pool

        with pool_instance(processes=self.workers_num, **self.pool_kwargs) as pool:
            return pool.starmap(func, args_list)

    def map(
        self,
        func: Callable[[T_tgt], T_map],
        arg_list: Iterable[T_tgt],
        start_method: Optional[Literal["spawn", "fork", "forkserver"]] = None,
    ) -> list[T_map]:
        """This function is a wrapper for map from multiprocessing.

        Args:
            func (Callable[[Iterable[T_tgt]], T_map]): Function to be mapped.
            arg_list (Iterable[Iterable[T_tgt]]): Arguments to be mapped.
            start_method (Optional[Literal["spawn", "fork", "forkserver"]], optional):
                Start method for multiprocessing. Defaults to None.
                If None, use the default start method of the system.

        Returns:
            list[T_map]: Results.
        """

        if self.workers_num == 1:
            return list(map(func, arg_list))

        pool_instance = get_context(start_method).Pool if start_method else Pool

        with pool_instance(processes=self.workers_num, **self.pool_kwargs) as pool:
            return pool.map(func, arg_list)

    def process_map(
        self,
        func: Callable[..., T_map],
        args_list: Iterable[Iterable[Any]],
        bar_format: str = "qurry-full",
        bar_ascii: str = "4squares",
        **kwargs,
    ) -> list[T_map]:
        """Call process_map from tqdm.
        This function is a wrapper for process_map from tqdm.
        But, it won't use `pool_kwargs` for they are different implementations
        with `multiprocessing.Pool` in this class.

        Args:
            func (Callable[[Any], T_map]): Function to be mapped.
            args (Iterable[Any]): Arguments to be mapped.
            bar_format (str, optional): Progress bar format. Defaults to "qurry-full".
            bar_ascii (str, optional): Progress bar ascii. Defaults to "4squares".
            **kwargs: Other arguments.

        Returns:
            list[T_map]: Results.
        """

        result_setup = default_setup(bar_format, bar_ascii)
        actual_bar_format = result_setup["bar_format"]
        actual_ascii = result_setup["ascii"]

        return process_map(
            func,
            *zip(*args_list),
            **kwargs,
            ascii=actual_ascii,
            bar_format=actual_bar_format,
            max_workers=self.workers_num,
        )
