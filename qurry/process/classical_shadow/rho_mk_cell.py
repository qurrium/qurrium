"""Post Processing - Classical Shadow - Rho M K Cell
(:mod:`qurry.process.classical_shadow.rho_mk_cell`)

"""

from typing import Literal, Union, Any
import numpy as np

from .unitary_set import U_M_MATRIX, OUTER_PRODUCT, IDENTITY
from ..availability import (
    availablility,
    default_postprocessing_backend,
    # PostProcessingBackendLabel,
)
from ..utils import counts_under_degree_pyrust

# from ..exceptions import (
#     PostProcessingRustImportError,
#     PostProcessingRustUnavailableWarning,
#     PostProcessingBackendDeprecatedWarning,
# )

# try:

#     from ...boorust import randomized  # type: ignore

#     purity_cell_2_rust_source = randomized.purity_cell_2_rust

#     RUST_AVAILABLE = True
#     FAILED_RUST_IMPORT = None
# except ImportError as err:
#     RUST_AVAILABLE = False
#     FAILED_RUST_IMPORT = err

#     def purity_cell_rust_source(*args, **kwargs):
#         """Dummy function for purity_cell_rust."""
#         raise PostProcessingRustImportError(
#             "Rust is not available, using python to calculate purity cell."
#         ) from FAILED_RUST_IMPORT


RUST_AVAILABLE = False
FAILED_RUST_IMPORT = None

BACKEND_AVAILABLE = availablility(
    "classical_shadow.rho_m_cell",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(RUST_AVAILABLE, False)


def rho_mk_cell_py(
    idx: int,
    single_counts: dict[str, int],
    nu_shadow_direction: dict[int, Union[Literal[0, 1, 2], int]],
    selected_classical_registers: list[int],
) -> tuple[
    int,
    dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
    dict[str, int],
    list[int],
]:
    r""":math:`\rho_{mk}` calculation for single cell.

    The matrix :math:`\rho_{mk}^{i} is calculated by the following equation,
    .. math::

        \rho_{mk}^{i} = \frac{3} U_{mi}^{\dagger} |b_k \rangle\langle b_k | U_{mi} - \mathbb{1}

    The matrix :math:`\rho_{mk}` is calculated by the following equation,
    .. math::

        \rho_{mk} = \bigotimes_{i=1}^{N_q} \rho_{mk}^{i}

    where :math:`N_q` is the number of qubits,

    Args:
        idx (int):
            Index of the cell (counts).
        single_counts (dict[str, int]):
            Counts measured by the single quantum circuit.
        nu_shadow_direction (dict[int, Union[Literal[0, 1, 2], int]]):
            The shadow direction of the unitary operators.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[
            int,
            dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]],
            dict[str, int],
            list[int],
        ]:
            Index, rho_mk, the sorted list of the selected qubits
    """

    # subsystem making
    num_classical_register = len(list(single_counts.keys())[0])
    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    single_counts_under_degree = counts_under_degree_pyrust(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )

    # core calculation
    rho_m_k_i_dict: dict[
        str, dict[int, np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.complex128]]]
    ] = {}
    rho_m_k_dict: dict[str, np.ndarray[tuple[int, int], np.dtype[np.complex128]]] = {}
    rho_m_k_counts_num: dict[str, int] = {}

    for bitstring, num_bitstring in single_counts_under_degree.items():
        rho_m_k_i_dict[bitstring] = {}
        rho_m_k_counts_num[bitstring] = num_bitstring
        for q_di, s_q in zip(selected_classical_registers_sorted, bitstring):
            rho_m_k_i_dict[bitstring][q_di] = (
                3
                * U_M_MATRIX[nu_shadow_direction[q_di]].conj().T
                @ OUTER_PRODUCT[s_q]
                @ U_M_MATRIX[nu_shadow_direction[q_di]]
            ) - IDENTITY

        tmp: Any = rho_m_k_i_dict[bitstring][selected_classical_registers_sorted[0]]
        for q_di in selected_classical_registers_sorted[1:]:
            tmp = np.kron(tmp, rho_m_k_i_dict[bitstring][q_di])
        rho_m_k_dict[bitstring] = tmp

    return idx, rho_m_k_dict, rho_m_k_counts_num, selected_classical_registers_sorted
