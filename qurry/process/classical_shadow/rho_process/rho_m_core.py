"""Post Processing - Classical Shadow - Rho Process - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_core`)

"""

from typing import Literal
from collections.abc import Iterable
import time
import numpy as np
import numpy.typing as npt

from .unitary_set import ShadowRandomBasis, ShadowBasisMethod, ShadowBasisType, DEFAULT_SHADOW_BASIS
from .rho_m_cell import rho_m_cell_precomputed, rho_m_cell_vectorized, RhoMCellMethod
from ..utils import spreadout
from ...utils import (
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
    BaseMethodEnum,
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
    selected_classical_registers: Iterable[int] | None = None,
    convert_to_single_shot: bool = False,
    rho_method: RhoMCellMethod = "numpy",
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
) -> tuple[list[npt.NDArray[np.complex128]], list[int], ShadowRandomBasis, float]:
    r"""Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int] | None, optional):
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
            It can be either "numpy" or "numpy_vectorized".

            - "numpy":
                Use Numpy to calculate the rho_m with precomputed values.
            - "numpy_vectorized":
                Use Numpy to calculate the rho_m with a vectorized workflow.
        shadow_basis (ShadowBasisType, optional):
            The shadow basis to use. Defaults to :data:`DEFAULT_SHADOW_BASIS`.

            Here are the built-in basis sets:
            - `RX_RY_RZ`:
                Uses :math:`R_X(\frac{\pi}{2})`, :math:`R_Y(-\frac{\pi}{2})`,
                and :math:`R_Z(0)` gates.
            - `H_H-Sdg_I`:
                Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`,
                and Identity gates.

    Returns:
        The dictionary of rho_m, the sorted list of the selected qubits,
        the shadow basis object, and calculation time.
    """

    shadow_basis_obj = ShadowBasisMethod.get_shadow_basis(shadow_basis)

    total_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )

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
            rho_m_cell_vectorized(bits_array, count_num, shadow_basis_obj)
            for bits_array, count_num in flatten_recount_list_vectorized
        ]  # where the bottleneck is

    else:
        rho_m_list = [
            rho_m_cell_precomputed(
                single_counts, random_unitary_array[idx], selected_clregs_sorted, shadow_basis_obj
            )
            for idx, single_counts in enumerate(counts_under_degree_list)
        ]

    taken = time.time() - begin

    return rho_m_list, selected_clregs_sorted, shadow_basis_obj, taken


class RhoMethod(BaseMethodEnum):
    """The method to use for the rho_m_core calculation.

    It can be either "multi_shots", "multi_shots_vectorized",
    "single_shots", or "single_shots_vectorized".

    For the "multi_shots_*" methods, the counts and random basis are used as is.
    For the "single_shots_*" methods, the counts and random basis are converted to single
    shot per snapshot for classical shadow post-processing.

    **Warning: Althought larger snapshots number means more accurate values.**
    **But if your shots number is large, this may significantly increase memory usage**
    **and require a lot of computing resource.**
    **In worst scenrio, this will break your computer.**
    **Please reconsider for performance.**

    - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
    - "multi_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow.

    - "single_shots": Use Numpy to calculate the rho_m with precomputed values
        with converted single shot counts.
    - "single_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow
        with converted single shot counts.

    Currently, "multi_shots" is the best option for performance.
    """

    MULTI_SHOTS = "multi_shots"
    """Use Numpy to calculate the rho_m with precomputed values."""
    MULTI_SHOTS_VECTORIZED = "multi_shots_vectorized"
    """Use Numpy to calculate the rho_m with a vectorized workflow."""
    SINGLE_SHOTS = "single_shots"
    """Use Numpy to calculate the rho_m with precomputed values 
    with converted single shot counts."""
    SINGLE_SHOTS_VECTORIZED = "single_shots_vectorized"
    """Use Numpy to calculate the rho_m with a vectorized workflow 
    with converted single shot counts."""

    @classmethod
    def get_default(cls):
        """Get the default method.

        Returns:
            The default enum member.
        """
        return cls.MULTI_SHOTS

    def is_single_method(self) -> bool:
        """Whether it is a single shot method.

        Returns:
            bool: True if it is a single shot method, False otherwise.
        """
        return "single" in self.value

    def is_multi_method(self) -> bool:
        """Whether it is a multi shot method.

        Returns:
            bool: True if it is a multi shot method, False otherwise.
        """
        return "multi" in self.value

    @classmethod
    def get_all_single_methods(cls) -> list[str]:
        """Get a list of all available single shot methods.

        Returns:
            list[str]: A list of single shot method names.
        """
        return [method.value for method in cls if method.is_single_method()]

    @classmethod
    def get_all_multi_methods(cls) -> list[str]:
        """Get a list of all available multi shot methods.

        Returns:
            list[str]: A list of multi shot method names.
        """
        return [method.value for method in cls if method.is_multi_method()]

    def whether_convert_and_rho_m_cell_method(self) -> tuple[bool, RhoMCellMethod]:
        """Whether to convert to single shot per snapshot for classical shadow
        post-processing and which rho_m_cell method to use.

        Returns:
            A tuple where the first element indicates if conversion is needed,
            and the second element is the corresponding RhoMCellMethod.
        """
        method_split_tuple = self.value.split("_shots", 1)
        convert_to_single_shot = "single" in method_split_tuple[0]
        rho_m_cell_method = (
            ("numpy" + method_split_tuple[1]) if len(method_split_tuple) > 1 else "numpy"
        )

        return convert_to_single_shot, rho_m_cell_method


RhoMethodType = RhoMethod | str
"""Type for rho_m_core method.

It can be either "multi_shots", "multi_shots_vectorized",
"single_shots", or "single_shots_vectorized".

For the "multi_shots_*" methods, the counts and random basis are used as is.
For the "single_shots_*" methods, the counts and random basis are converted to single
shot per snapshot for classical shadow post-processing.

**Warning: Althought larger snapshots number means more accurate values.**
**But if your shots number is large, this may significantly increase memory usage**
**and require a lot of computing resource.**
**In worst scenrio, this will break your computer.**
**Please reconsider for performance.**

- "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
- "multi_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow.

- "single_shots": Use Numpy to calculate the rho_m with precomputed values with
    converted single shot counts.
- "single_shots_vectorized": Use Numpy to calculate the rho_m with a vectorized workflow
    with converted single shot counts.

Currently, "multi_shots" is the best option for performance.
"""

DEFAULT_RHO_METHOD: RhoMethod = RhoMethod.get_default()
"""The default method for rho_m_core.

Currently, "multi_shots" is the best option for performance.
"""


def rho_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[Literal[0, 1, 2] | int]],
    selected_classical_registers: Iterable[int] | None = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
) -> tuple[list[npt.NDArray[np.complex128]], list[int], ShadowRandomBasis, float]:
    r"""Rho M Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Literal[0, 1, 2] | int]],):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int] | None, optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        rho_method (RhoMethodType, optional):
            It can be either "multi_shots", "multi_shots_vectorized",
            "single_shots", or "single_shots_vectorized".

            For the "multi_shots_*" methods, the counts and random basis are used as is.
            For the "single_shots_*" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Althought larger snapshots number means more accurate values.**
            **But if your shots number is large,**
            **this may significantly increase memory usage**
            **and require a lot of computing resource.**
            **In worst scenrio, this will break your computer.**
            **Please reconsider for performance.**

            - "multi_shots": Use Numpy to calculate the rho_m with precomputed values.
            - "multi_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow.

            - "single_shots": Use Numpy to calculate the rho_m
                with precomputed values with converted single shot counts.
            - "single_shots_vectorized": Use Numpy to calculate the rho_m
                with a vectorized workflow with converted single shot counts.

            Currently, "multi_shots" is the best option for performance.
            Default to DEFAULT_RHO_METHOD, which is "multi_shots".
        shadow_basis (ShadowBasisType, optional):
            The shadow basis to use. Defaults to :data:`DEFAULT_SHADOW_BASIS`.

            Here are the built-in basis sets:
            - `RX_RY_RZ`:
                Uses :math:`R_X(\frac{\pi}{2})`, :math:`R_Y(-\frac{\pi}{2})`,
                and :math:`R_Z(0)` gates.
            - `H_H-Sdg_I`:
                Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`,
                and Identity gates.

    Returns:
        The dictionary of rho_m, the sorted list of the selected qubits,
        the shadow basis object, and calculation time.
    """

    if isinstance(rho_method, str):
        rho_method = RhoMethod(rho_method)
    convert_to_single_shot, rho_m_core_method = rho_method.whether_convert_and_rho_m_cell_method()

    return rho_m_core_py(
        shots=shots,
        counts=counts,
        random_unitary_array=random_unitary_array,
        selected_classical_registers=selected_classical_registers,
        convert_to_single_shot=convert_to_single_shot,
        rho_method=rho_m_core_method,
        shadow_basis=shadow_basis,
    )


def mean_rho_core(
    rho_m_list: list[npt.NDArray[np.complex128]], selected_classical_registers_sorted: list[int]
) -> npt.NDArray[np.complex128]:
    """Calculate the expectation value of Rho.

    Args:
        rho_m_list (list[npt.NDArray[np.complex128]]):
            The list of rho_m.
        selected_classical_registers_sorted (list[int]):
            The list of the selected_classical_registers.

    Returns:
        npt.NDArray[np.complex128]: The expectation value of Rho.
    """

    expect_rho: npt.NDArray[np.complex128] = np.sum(rho_m_list, axis=0, dtype=np.complex128)
    assert expect_rho.shape == (2 ** len(selected_classical_registers_sorted),) * 2, (
        f"The shape of expect_rho: {expect_rho.shape} "
        + f"and the shape of rho_m_list: {rho_m_list[0].shape} are different."
    )
    expect_rho /= len(rho_m_list)

    return expect_rho
