"""Post Processing - Classical Shadow - Matrix Calculation
(:mod:`qurry.process.classical_shadow.matrix_calcution`)

"""

from typing import Iterable, Literal, Callable, Union
import warnings
import functools as ft
import numpy as np

from .unitary_set import PRECOMPUTED_RHO_M_K_I, PRECOMPUTED_RHO_M_K_I_2
from ..availability import availablility
from ..exceptions import (
    PostProcessingThirdPartyImportError,
    PostProcessingThirdPartyUnavailableWarning,
)

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    # =========================================================
    # This is required to handle the complex128 dtype in JAX.
    # Or the result of JAX will be not same as Numpy.
    # =========================================================

    # kronocker product calculation
    def rho_mki_kronecker_product_jax(
        key_list_of_precomputed: list[tuple[int, str]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        r"""Kronecker product for :math:`\rho_{mki}` by JAX.

        Args:
            key_list_of_precomputed (list[tuple[int, str]]):
                The list of the keys of the precomputed :math:`\rho_{mki}`.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
                The Kronecker product of the :math:`\rho_{mki}`.
        """
        return np.array(
            ft.reduce(jnp.kron, [PRECOMPUTED_RHO_M_K_I[key] for key in key_list_of_precomputed])
        )

    def rho_mki_kronecker_product_jax_2(
        key_list_of_precomputed: Iterable[int],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        r"""Kronecker product for :math:`\rho_{mki}` by JAX.

        Args:
            key_list_of_precomputed (Iterable[int]):
                The list of the keys of the precomputed :math:`\rho_{mki}`.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
                The Kronecker product of the :math:`\rho_{mki}`.
        """
        return np.array(
            ft.reduce(jnp.kron, [PRECOMPUTED_RHO_M_K_I_2[key] for key in key_list_of_precomputed])
        )

    # rho_m calculation
    def random_unitary_um_to_nu_dir_array_under_degree_jax(
        random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
        selected_classical_registers_sorted: list[int],
    ) -> jnp.ndarray:
        """Convert the random unitary um to nu_dir_array
        with selected_classical_registers.

        Args:
            random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
                The shadow direction of the unitary operators.
            selected_classical_registers_sorted (list[int]):
                The list of the selected_classical_registers.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.int32]]: The nu_dir_array.
        """

        random_unitary_ids_array = jnp.array([list(v.values()) for v in random_unitary_um.values()])
        random_unitary_ids_array_under_degree = random_unitary_ids_array[
            :, selected_classical_registers_sorted
        ]
        return random_unitary_ids_array_under_degree

    def process_single_count_jax(
        nu_dir_array: np.ndarray[tuple[int], np.dtype[np.int32]],
        single_counts: dict[str, int],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """Process a single count by JAX.

        Args:
            nu_dir_array (np.ndarray[tuple[int], np.dtype[np.int32]]):
                The shadow direction of the unitary operators.
            single_counts (dict[str, int]):
                The single count.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]: The rho_m.
        """

        keys = list(single_counts.keys())
        values = jnp.array(list(single_counts.values()))
        total_count = sum(values)

        keys_int_array = jnp.array([list(map(int, k)) for k in keys], dtype=np.int32)
        nu_expanded = jnp.broadcast_to(nu_dir_array, (len(keys), len(nu_dir_array)))
        lookup_keys = nu_expanded * 10 + keys_int_array

        rho_m_k_unweighted = jnp.array(
            [rho_mki_kronecker_product_jax_2(kl) for kl in np.array(lookup_keys)]
        )

        rho_m_k_weighted = np.array([m * v for m, v in zip(rho_m_k_unweighted, values)])
        rho_m = rho_m_k_weighted.sum(axis=0)

        return np.array(rho_m / total_count, dtype=np.complex128)

    # trace summation calculation
    def all_trace_rho_by_einsum_aij_bji_to_ab_jax(
        rho_m_array: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ) -> np.complex128:
        """The trace of Rho by einsum_aij_bji_to_ab by JAX.

        This is the fastest implementation to calculate the trace of Rho.

        Args:
            rho_m_array (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                The Rho M array.
        Returns:
            np.complex128: The trace of Rho.
        """
        len_rho_m_array = len(rho_m_array)
        trace_matrix = jnp.einsum("aij,bji -> ab", rho_m_array, rho_m_array)

        mask = np.ones(trace_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)

        sum_off_diagonal = trace_matrix[mask].sum()
        return np.complex128(sum_off_diagonal / (len_rho_m_array * (len_rho_m_array - 1)))

    JAX_AVAILABLE = True
    FAILED_JAX_IMPORT = None
except ImportError as err:
    JAX_AVAILABLE = False
    FAILED_JAX_IMPORT = err

    # kronecker product calculation
    def rho_mki_kronecker_product_jax(
        key_list_of_precomputed: list[tuple[int, str]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        r"""Kronecker product for :math:`\rho_{mki}` by JAX.

        Args:
            key_list_of_precomputed (list[tuple[int, str]]):
                The list of the keys of the precomputed :math:`\rho_{mki}`.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
                The Kronecker product of the :math:`\rho_{mki}`.
        """
        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate Kronecker product."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    def rho_mki_kronecker_product_jax_2(
        key_list_of_precomputed: Iterable[int],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        r"""Kronecker product for :math:`\rho_{mki}` by JAX.

        Args:
            key_list_of_precomputed (Iterable[int]):
                The list of the keys of the precomputed :math:`\rho_{mki}`.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
                The Kronecker product of the :math:`\rho_{mki}`.
        """
        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate Kronecker product."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    # rho_m calculation
    def random_unitary_um_to_nu_dir_array_under_degree_jax(
        random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
        selected_classical_registers_sorted: list[int],
    ) -> jnp.ndarray:
        """Convert the random unitary um to nu_dir_array
        with selected_classical_registers.

        Args:
            random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
                The shadow direction of the unitary operators.
            selected_classical_registers_sorted (list[int]):
                The list of the selected_classical_registers.

        Returns:
            jnp.ndarray: The nu_dir_array.
        """

        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate Kronecker product."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    def process_single_count_jax(
        nu_dir_array: np.ndarray[tuple[int], np.dtype[np.int32]],
        single_counts: dict[str, int],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        """Process a single count by JAX.

        Args:
            nu_dir_array (np.ndarray[tuple[int], np.dtype[np.int32]]):
                The shadow direction of the unitary operators.
            single_counts (dict[str, int]):
                The single count.

        Returns:
            jnp.ndarray: The rho_m.
        """

        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate Kronecker product."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    # trace summation calculation
    def all_trace_rho_by_einsum_aij_bji_to_ab_jax(
        rho_m_array: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ) -> np.complex128:
        """The trace of Rho by einsum_aij_bji_to_ab by JAX.

        This is the fastest implementation to calculate the trace of Rho.

        Args:
            rho_m_array (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                The Rho M array.
        Returns:
            np.complex128: The trace of Rho.
        """
        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate Kronecker product."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT


BACKEND_AVAILABLE = availablility(
    "classical_shadow.array_process",
    [
        ("jax", JAX_AVAILABLE, FAILED_JAX_IMPORT),
    ],
)
ClassicalShadowPythonMethod = Literal["jax", "numpy"]
"""The method to use for the calculation of classical shadow.
It can be either "jax" or "numpy".
- "jax": Use JAX to calculate the Kronecker product.
- "numpy": Use Numpy to calculate the Kronecker product.
"""
DEFAULT_PYTHON_METHOD: ClassicalShadowPythonMethod = "jax" if JAX_AVAILABLE else "numpy"
"""The default backend to use for the calculation of classical shadow.

It can be either "jax" or "numpy".
- "jax": Use JAX to calculate the Kronecker product.
- "numpy": Use Numpy to calculate the Kronecker product.
"""


# kronecker product calculation
def rho_mki_kronecker_product_numpy(
    key_list_of_precomputed: list[tuple[int, str]],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    r"""Kronecker product for :math:`\rho_{mki}` by Numpy.

    Args:
        key_list_of_precomputed (list[tuple[int, str]]):
            The list of the keys of the precomputed :math:`\rho_{mki}`.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
            The Kronecker product of the :math:`\rho_{mki}`.
    """
    return ft.reduce(np.kron, [PRECOMPUTED_RHO_M_K_I[key] for key in key_list_of_precomputed])


def rho_mki_kronecker_product_numpy_2(
    key_list_of_precomputed: Iterable[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    r"""Kronecker product for :math:`\rho_{mki}` by Numpy.

    Args:
        key_list_of_precomputed (Iterable[int]):
            The list of the keys of the precomputed :math:`\rho_{mki}`.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
            The Kronecker product of the :math:`\rho_{mki}`.
    """
    return ft.reduce(np.kron, [PRECOMPUTED_RHO_M_K_I_2[key] for key in key_list_of_precomputed])


def select_rho_mki_kronecker_product(
    method: ClassicalShadowPythonMethod = DEFAULT_PYTHON_METHOD,
) -> Callable[[list[tuple[int, str]]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
    r"""Select the method for Kronecker product for :math:`\rho_{mki}`.

    Args:
        method (ClassicalShadowPythonMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_PYTHON_METHOD.

    Returns:
        Callable[[list[tuple[int, str]]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
            The function to calculate the Kronecker product of the :math:`\rho_{mki}`.
    """
    if method == "jax":
        if JAX_AVAILABLE:
            return rho_mki_kronecker_product_jax
        warnings.warn(
            "JAX is not available, using numpy to calculate Kronecker product.",
            PostProcessingThirdPartyUnavailableWarning,
        )
        method = "numpy"
    if method != "numpy":
        raise ValueError(f"Invalid backend: {method}")
    return rho_mki_kronecker_product_numpy


def select_rho_mki_kronecker_product_2(
    method: ClassicalShadowPythonMethod = DEFAULT_PYTHON_METHOD,
) -> Callable[[Iterable[int]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
    r"""Select the method for Kronecker product for :math:`\rho_{mki}`.

    Args:
        method (ClassicalShadowPythonMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_PYTHON_METHOD.

    Returns:
        Callable[[list[tuple[int, str]]], np.ndarray[tuple[int, int], np.dtype[np.complex128]]]:
            The function to calculate the Kronecker product of the :math:`\rho_{mki}`.
    """
    if method == "jax":
        if JAX_AVAILABLE:
            return rho_mki_kronecker_product_jax_2
        warnings.warn(
            "JAX is not available, using numpy to calculate Kronecker product.",
            PostProcessingThirdPartyUnavailableWarning,
        )
        method = "numpy"
    if method != "numpy":
        raise ValueError(f"Invalid backend: {method}")
    return rho_mki_kronecker_product_numpy_2


# rho_m calculation
def random_unitary_um_to_nu_dir_array_under_degree_numpy(
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers_sorted: list[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
    """Convert the random unitary um to nu_dir_array
    with selected_classical_registers.

    Args:
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers_sorted (list[int]):
            The list of the selected_classical_registers.

    Returns:
        np.ndarray[tuple[int], np.dtype[np.int32]]: The nu_dir_array.
    """

    random_unitary_ids_array = np.array([list(v.values()) for v in random_unitary_um.values()])
    random_unitary_ids_array_under_degree = random_unitary_ids_array[
        :, selected_classical_registers_sorted
    ]
    return random_unitary_ids_array_under_degree


def select_random_unitary_um_to_nu_dir_array_under_degree(
    method: ClassicalShadowPythonMethod = DEFAULT_PYTHON_METHOD,
) -> Callable[
    [dict[int, dict[int, Union[Literal[0, 1, 2], int]]], list[int]],
    Union[np.ndarray[tuple[int, int], np.dtype[np.int32]], jnp.ndarray],
]:
    """Select the method for converting random unitary um to nu_dir_array.

    Args:
        method (ClassicalShadowPythonMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_PYTHON_METHOD.

    Returns:
        Callable[
            [dict[int, dict[int, Union[Literal[0, 1, 2], int]]], list[int]],
            Union[np.ndarray[tuple[int, int], np.dtype[np.int32]], jnp.ndarray]
        ]: The function to convert random unitary um to nu_dir_array.
    """
    if method == "jax":
        if JAX_AVAILABLE:
            return random_unitary_um_to_nu_dir_array_under_degree_jax
        warnings.warn(
            "JAX is not available, using numpy to calculate Kronecker product.",
            PostProcessingThirdPartyUnavailableWarning,
        )
        method = "numpy"
    if method != "numpy":
        raise ValueError(f"Invalid backend: {method}")
    return random_unitary_um_to_nu_dir_array_under_degree_numpy


def process_single_count_numpy(
    nu_dir_array: np.ndarray[tuple[int], np.dtype[np.int32]],
    single_counts: dict[str, int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    """Process a single count by Numpy.

    Args:
        nu_dir_array (np.ndarray[tuple[int], np.dtype[np.int32]]):
            The shadow direction of the unitary operators.
        single_counts (dict[str, int]):
            The single count.

    Returns:
        np.ndarray[tuple[int, int], np.dtype[np.complex128]]: The rho_m.
    """

    keys = list(single_counts.keys())
    values = list(single_counts.values())
    total_count = sum(values)

    keys_int_array = np.array([list(map(int, k)) for k in keys], dtype=np.int32)
    nu_expanded = np.broadcast_to(nu_dir_array, (len(keys), len(nu_dir_array)))
    lookup_keys = nu_expanded * 10 + keys_int_array

    rho_m_k_unweighted = [rho_mki_kronecker_product_numpy_2(kl) for kl in lookup_keys]

    rho_m_k_weighted = np.array([m * v for m, v in zip(rho_m_k_unweighted, values)])
    rho_m = rho_m_k_weighted.sum(axis=0, dtype=np.complex128)

    return rho_m / total_count


def select_process_single_count(
    method: ClassicalShadowPythonMethod = DEFAULT_PYTHON_METHOD,
) -> Callable[
    [np.ndarray[tuple[int], np.dtype[np.int32]], dict[str, int]],
    np.ndarray[tuple[int, int], np.dtype[np.complex128]],
]:
    """Select the method for processing a single count.

    Args:
        method (ClassicalShadowPythonMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_PYTHON_METHOD.

    Returns:
        Callable[
            [np.ndarray[tuple[int], np.dtype[np.int32]], dict[str, int]],
            np.ndarray[tuple[int, int], np.dtype[np.complex128]]
        ]:
            The function to process a single count.
    """
    if method == "jax":
        if JAX_AVAILABLE:
            return process_single_count_jax
        warnings.warn(
            "JAX is not available, using numpy to calculate Kronecker product.",
            PostProcessingThirdPartyUnavailableWarning,
        )
        method = "numpy"
    if method != "numpy":
        raise ValueError(f"Invalid backend: {method}")
    return process_single_count_numpy


# single trace calculation
def single_trace_rho_by_trace_of_matmul(
    rho_m1_and_rho_m2: tuple[
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    ],
) -> np.complex128:
    """The single trace of Rho by trace of matmul.

    Args:
        rho_m1_and_rho_m2 (tuple): The tuple of rho_m1 and rho_m2.

    Returns:
        np.complex128: The trace of Rho.
    """
    rho_m1, rho_m2 = rho_m1_and_rho_m2
    return np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))


def single_trace_rho_by_einsum_ij_ji(
    rho_m1_and_rho_m2: tuple[
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    ],
) -> np.complex128:
    """The single trace of Rho by einsum_ij_ji by Numpy.

    Args:
        rho_m1_and_rho_m2 (tuple): The tuple of rho_m1 and rho_m2.

    Returns:
        np.complex128: The trace of Rho.
    """
    rho_m1, rho_m2 = rho_m1_and_rho_m2
    return np.einsum("ij,ji", rho_m1, rho_m2) + np.einsum("ij,ji", rho_m2, rho_m1)


SingleTraceRhoMethod = Union[
    Literal[
        "trace_of_matmul",
        "quick_trace_of_matmul",
        "einsum_ij_ji",
    ],
    str,
]
"""The method to calculate the trace of single Rho square.

- "trace_of_matmul": Use 
    np.trace(np.matmul(rho_m1, rho_m2)) to calculate the trace.
- "quick_trace_of_matmul" or "einsum_ij_ji": 
    Use np.einsum("ij,ji", rho_m1, rho_m2) to calculate the trace.
"""


def select_single_trace_rho_method(
    method: SingleTraceRhoMethod = "quick_trace_of_matmul",
) -> Callable[
    [
        tuple[
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        ],
    ],
    Union[np.complex128, jnp.ndarray],
]:
    """Select the method to calculate the trace of Rho square.

    Args:
        method (str): The method to use for the calculation.

    Returns:
        Callable[[tuple], np.complex128]: The function to calculate the trace of Rho.
    """
    if method == "trace_of_matmul":
        return single_trace_rho_by_trace_of_matmul

    if method in ("quick_trace_of_matmul", "einsum_ij_ji"):
        return single_trace_rho_by_einsum_ij_ji

    raise ValueError(f"Invalid method: {method}")


# trace summation calculation
def all_trace_rho_by_einsum_aij_bji_to_ab_numpy(
    rho_m_array: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
) -> np.complex128:
    """The trace of Rho by einsum_aij_bji_to_ab.

    This is the fastest implementation to calculate the trace of Rho.

    Args:
        rho_m_array (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
            The Rho M array.
    Returns:
        np.complex128: The trace of Rho.
    """
    len_rho_m_array = len(rho_m_array)
    trace_matrix = np.einsum("aij,bji -> ab", rho_m_array, rho_m_array)

    mask = np.ones(trace_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)

    sum_off_diagonal = trace_matrix[mask].sum()
    return sum_off_diagonal / (len_rho_m_array * (len_rho_m_array - 1))


AllTraceRhoMethod = Union[Literal["einsum_aij_bji_to_ab_numpy", "einsum_aij_bji_to_ab_jax"], str]
"""The method to calculate the all trace of Rho square.

- "einsum_aij_bji_to_ab_numpy":
    Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho 
    if JAX is not available.
- "einsum_aij_bji_to_ab_jax":
    Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho.
"""
DEFAULT_ALL_TRACE_RHO_METHOD: AllTraceRhoMethod = (
    "einsum_aij_bji_to_ab_jax" if JAX_AVAILABLE else "einsum_aij_bji_to_ab_numpy"
)


def select_all_trace_rho_by_einsum_aij_bji_to_ab(
    method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
) -> Callable[
    [np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]],
    np.complex128,
]:
    """Select the method to calculate the trace of Rho square.

    Args:
        method (AllTraceRhoMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_ALL_TRACE_RHO_METHOD.
            It can be either "einsum_aij_bji_to_ab_numpy" or "einsum_aij_bji_to_ab_jax".
            - "einsum_aij_bji_to_ab_numpy":
                Use np.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            - "einsum_aij_bji_to_ab_jax":
                Use jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list) to calculate the trace.
            This is the fastest implementation to calculate the trace of Rho.

    Returns:
        Callable[[np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]], np.complex128]:
            The function to calculate the trace of Rho.
    """
    if method == "einsum_aij_bji_to_ab_jax":
        if JAX_AVAILABLE:
            return all_trace_rho_by_einsum_aij_bji_to_ab_jax
        warnings.warn(
            "JAX is not available, using numpy to calculate all trace.",
            PostProcessingThirdPartyUnavailableWarning,
        )
    if method != "einsum_aij_bji_to_ab_numpy":
        raise ValueError(f"Invalid backend: {method}")
    return all_trace_rho_by_einsum_aij_bji_to_ab_numpy
