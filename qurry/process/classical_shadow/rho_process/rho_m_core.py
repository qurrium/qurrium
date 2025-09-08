"""Post Processing - Classical Shadow - Rho Process - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_core`)

"""

import time
from typing import Literal, Union, Iterable, Optional
import numpy as np

from .rho_m_cell import (
    rho_m_cell_prototype,
    rho_m_cell_precomputed,
    rho_m_cell_vectorized,
    RhoMCellMethod,
)
from ..utils import spreadout, check_random_basis_array
from ...utils import (
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
)
from ...availability import availablility


BACKEND_AVAILABLE = availablility(
    "classical_shadow.rho_process", [("Rust", True, None), ("Numpy", True, None)]
)
"""The availability of backends for classical shadow rho process.

Rust for counts processing in the rho method of "numpy_vectorized" 
from :const:`~qurry.process.classical_shadow.rho_process.rho_m_cell.RhoMCellMethod`,
or "multi_shots_vectorized" and "single_shots_vectorized"
from :const:`RhoMethod`, which is the option the function
:func:`~qurry.process.classical_shadow.rho_process.rho_m_cell.rho_m_cell_vectorized`
"""


def rho_m_core_py(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[int]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    convert_to_single_shot: bool = False,
    rho_method: RhoMCellMethod = "numpy",
) -> tuple[
    list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    list[int],
    float,
]:
    """Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        convert_to_single_shot (bool, optional):
            Whether to convert the counts and the random basis from multiple shots
            to single shot per snapshot for classical shadow post-processing.
            **Warning: If your shots number is large. Please reconsider for performance.**
            **This may significantly increase memory usage and break your computer.**
            Default to False.
        rho_method (RhoMCoreMethod, optional):
            The method to use for the calculation. Defaults to "numpy".
            It can be either "numpy_proto", "numpy", or "numpy_vectorized".

            - "numpy_proto":
                Use Numpy to calculate the rho_m.
            - "numpy":
                Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_vectorized":
                Use Numpy to calculate the rho_m with a vectorized workflow.

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The dictionary of rho_m, the sorted list of the selected qubits, and calculation time.
    """

    total_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )
    check_random_basis_array(random_unitary_array, len(counts), total_system_size)

    if convert_to_single_shot:
        shots, counts, random_unitary_array = spreadout(shots, counts, random_unitary_array)

    begin = time.time()

    selected_clregs_sorted = sorted(selected_classical_registers, reverse=True)
    counts_under_degree_list = counts_list_recount_pyrust(
        counts,
        num_classical_register=total_system_size,
        selected_classical_registers=selected_clregs_sorted,
    )

    if rho_method == "numpy_vectorized":

        flatten_recount_list_vectorized = rho_m_flatten_counts_list_vectorize_pyrust(
            counts_under_degree_list, random_unitary_array, selected_clregs_sorted
        )  # Even parallel is not needed
        rho_m_list = [
            rho_m_cell_vectorized(bits_array, count_num)
            for bits_array, count_num in flatten_recount_list_vectorized
        ]  # where the bottleneck is

    else:
        cell_calculation_method = (
            rho_m_cell_precomputed if rho_method == "numpy" else rho_m_cell_prototype
        )
        rho_m_list = [
            cell_calculation_method(
                single_counts, random_unitary_array[idx], selected_clregs_sorted
            )
            for idx, single_counts in enumerate(counts_under_degree_list)
        ]

    taken = time.time() - begin

    return rho_m_list, selected_clregs_sorted, taken


RhoMethod = Union[
    Literal[
        "multi_shots_proto",
        "multi_shots",
        "multi_shots_vectorized",
        "single_shots_proto",
        "single_shots",
        "single_shots_vectorized",
    ],
    str,
]
"""Type for rho_m_core method.

It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
"single_shots_proto", "single_shots", or "single_shots_vectorized".

For the "multi_shots_*" methods, the counts and random basis are used as is.
For the "single_shots_*" methods, the counts and random basis are converted to single
shot per snapshot for classical shadow post-processing.

**Warning: Althought larger snapshots number means more accurate values.**
**But if your shots number is large, this may significantly increase memory usage**
**and require a lot of computing resource.**
**In worst scenrio, this will break your computer.**
**Please reconsider for performance.**

- "multi_shots_proto": Use Numpy to calculate the rho_m.
- "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
- "multi_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow.

- "single_shots_proto": Use Numpy to calculate the rho_m with converted single shot counts.
- "single_shots": Use Numpy to calculate the rho_m with precomputed values with
    converted single shot counts.
- "single_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow
    with converted single shot counts.

Currently, "multi_shots" is the best option for performance.
"""

DEFAULT_RHO_METHOD: RhoMethod = "multi_shots"
"""The default method for rho_m_core.

Currently, "multi_shots" is the best option for performance.
"""


def validate_rho_m_core_method(method: RhoMethod) -> tuple[bool, RhoMCellMethod]:
    """Validate the rho_m_core method.

    Args:
        method (RhoMCoreMethod): The method to validate.

    Returns:
        A tuple where the first element indicates if
        conversion is needed, and the second element is the corresponding RhoMCellMethod.

    Raises:
        ValueError: If the method is not valid.
    """

    if method not in [
        "multi_shots_proto",
        "multi_shots",
        "multi_shots_vectorized",
        "single_shots_proto",
        "single_shots",
        "single_shots_vectorized",
    ]:
        raise ValueError(
            f"Unknown rho_method: {method}. "
            + "Available methods are: 'multi_shots_proto', 'multi_shots', "
            + "'multi_shots_vectorized', 'single_shots_proto', 'single_shots', "
            + "'single_shots_vectorized'."
        )

    method_split_tuple = method.split("_shots", 1)
    convert_to_single_shot = "single" in method_split_tuple[0]
    rho_m_cell_method = (
        ("numpy" + method_split_tuple[1]) if len(method_split_tuple) > 1 else "numpy"
    )

    return convert_to_single_shot, rho_m_cell_method


def rho_m_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    rho_method: RhoMethod = DEFAULT_RHO_METHOD,
) -> tuple[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]], list[int], float]:
    """Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]],):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        rho_method (RhoMethod, optional):
            It can be either "multi_shots_proto", "multi_shots", "multi_shots_vectorized",
            "single_shots_proto", "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots_proto": Use Numpy to calculate the rho_m.
            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots_proto": Use Numpy to calculate the rho_m
                with converted single shot counts.
            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".

    Returns:
        tuple[
            list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            list[int],
            float
        ]:
            The dictionary of rho_m, the sorted list of the selected qubits, and calculation time.
    """

    convert_to_single_shot, rho_m_core_method = validate_rho_m_core_method(rho_method)

    return rho_m_core_py(
        shots=shots,
        counts=counts,
        random_unitary_array=random_unitary_array,
        selected_classical_registers=selected_classical_registers,
        convert_to_single_shot=convert_to_single_shot,
        rho_method=rho_m_core_method,
    )
