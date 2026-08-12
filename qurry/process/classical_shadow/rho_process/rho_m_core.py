"""Post Processing - Classical Shadow - Rho Process - Rho M Core
(:mod:`qurry.process.classical_shadow.rho_process.rho_m_core`)

"""

from collections.abc import Iterable
import time
import numpy as np
import numpy.typing as npt

from .unitary_set import ShadowRandomBasis, ShadowBasisMethod, ShadowBasisType, DEFAULT_SHADOW_BASIS
from .rho_m_cell import rho_m_cell_precomputed
from ..utils import spreadout
from ...utils import (
    BaseMethodEnum,
    counts_list_recount_pyrust,
    shot_counts_selected_clreg_checker_pyrust,
)


class RhoMethod(BaseMethodEnum):
    """Whether to use the multi_shots or single_shots method for rho_m_cell calculation.
    For the "multi_shots" methods, the counts and random basis are used as is.
    For the "single_shots" methods, the counts and random basis are
    converted to single shot per snapshot for classical shadow post-processing.

    **Warning: Although larger snapshots number means more accurate values.**
    **But if your shots number is large, this may significantly increase memory usage,**
    **require a lot of computing resource, and may run out of memory.**
    **Please consider carefully for performance.**

    Default to "multi_shots".
    """

    MULTI_SHOTS = "multi_shots"
    """Use Numpy to calculate the rho_m with precomputed values."""
    SINGLE_SHOTS = "single_shots"
    """Use Numpy to calculate the rho_m with precomputed values 
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


DEFAULT_RHO_METHOD = RhoMethod.get_default()
"""Whether to use the multi_shots or single_shots method for rho_m_cell calculation.
For the "multi_shots" methods, the counts and random basis are used as is.
For the "single_shots" methods, the counts and random basis are
converted to single shot per snapshot for classical shadow post-processing.

**Warning: Although larger snapshots number means more accurate values.**
**But if your shots number is large, this may significantly increase memory usage,**
**require a lot of computing resource, and may run out of memory.**
**Please consider carefully for performance.**

Default to "multi_shots".
"""

RhoMethodType = RhoMethod | str
"""Whether to use the multi_shots or single_shots method for rho_m_cell calculation.
For the "multi_shots" methods, the counts and random basis are used as is.
For the "single_shots" methods, the counts and random basis are
converted to single shot per snapshot for classical shadow post-processing.
"""


def rho_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[int]],
    selected_classical_registers: Iterable[int] | None = None,
    rho_method: RhoMethodType = DEFAULT_RHO_METHOD,
    shadow_basis: ShadowBasisType = DEFAULT_SHADOW_BASIS,
    use_projecter: bool = False,
) -> tuple[list[npt.NDArray[np.complex128]], list[int], ShadowRandomBasis, float]:
    r"""Rho M Cell Core calculation.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_unitary_array (list[list[int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (Iterable[int] | None, optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        rho_method (RhoMethodType, optional):
            For the "multi_shots" methods, the counts and random basis are used as is.
            For the "single_shots" methods, the counts and random basis are
            converted to single shot per snapshot for classical shadow post-processing.

            **Warning: Although larger snapshots number means more accurate values.**
            **But if your shots number is large, this may significantly increase memory usage,**
            **require a lot of computing resource, and may run out of memory.**
            **Please consider carefully for performance.**

            Default to "multi_shots".
        shadow_basis (ShadowBasisType, optional):
            The shadow basis to use. Defaults to :data:`DEFAULT_SHADOW_BASIS`.

            Here are the built-in basis sets:
            - `RY_RX_RZ`:
                Uses :math:`R_Y(-\frac{\pi}{2})`, :math:`R_X(\frac{\pi}{2})`,
                and :math:`R_Z(0)` gates.
            - `H_H-Sdg_I`:
                Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`,
                and Identity gates.
        use_projecter (bool):
            Use the projecter :math:`P_m` instead of the precomputed :math:`\rho_m`.
            Refer to
            :func:`qurry.process.classical_shadow.rho_process.rho_m_cell.rho_m_cell_precomputed`
            for more details.
            Default is False, which means using the precomputed :math:`\rho_{mk}^{i}`.

    Returns:
        The dictionary of :math:`\rho_{m}` or :math:`P_m`
        depending on the value of :attr:`use_projecter`,
        the sorted list of the selected qubits,
        the shadow basis object, and calculation time.
    """

    if isinstance(rho_method, str):
        rho_method = RhoMethod.from_string(rho_method)
    shadow_basis_obj = ShadowBasisMethod.get_shadow_basis(shadow_basis)

    total_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )
    if rho_method == RhoMethod.SINGLE_SHOTS:
        shots, counts, random_unitary_array = spreadout(shots, counts, random_unitary_array)

    begin = time.time()

    selected_clregs_sorted = sorted(selected_classical_registers, reverse=True)
    counts_under_degree_list = counts_list_recount_pyrust(
        counts,
        num_classical_register=total_system_size,
        selected_classical_registers=selected_clregs_sorted,
    )

    return (
        [
            rho_m_cell_precomputed(
                single_counts,
                random_unitary_array[idx],
                selected_clregs_sorted,
                shadow_basis_obj,
                use_projecter=use_projecter,
            )
            for idx, single_counts in enumerate(counts_under_degree_list)
        ],
        selected_clregs_sorted,
        shadow_basis_obj,
        time.time() - begin,
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

    expect_rho = np.sum(rho_m_list, axis=0, dtype=np.complex128)
    assert expect_rho.shape == (2 ** len(selected_classical_registers_sorted),) * 2, (
        f"The shape of expect_rho: {expect_rho.shape} "
        + f"and the shape of rho_m_list: {rho_m_list[0].shape} are different."
    )

    return expect_rho / len(rho_m_list)
