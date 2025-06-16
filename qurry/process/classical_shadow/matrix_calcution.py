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

    def prediction_einsum_aij_bji_to_ab_jax(
        given_operators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
        estimators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ) -> tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
        """Calculate the prediction of given operators by einsum_aij_bji_to_ab_jax.

            Args:
                given_operators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                    The given operators.
                estimators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                    The estimators.

        Returns:
            tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
                A tuple containing:
                - A list of median values for each given operator.
                - A list of the corresponding median estimators for each given operator.
        """
        candidate_esitmators_foreach_given_operator = jnp.einsum(
            "aij,bji->ab", given_operators, estimators
        )
        median_foreach_given_operator = np.median(
            candidate_esitmators_foreach_given_operator, axis=1
        )
        median_location_given_operator = np.argmin(
            np.abs(
                candidate_esitmators_foreach_given_operator - median_foreach_given_operator[:, None]
            ),
            axis=1,
        )

        return list(median_foreach_given_operator), [
            np.array(candidate_esitmators_foreach_given_operator[i, j], dtype=np.complex128)
            for i, j in enumerate(median_location_given_operator)
        ]  # type: ignore

    def set_cpu_only():
        """Set JAX to use CPU only."""
        if not jax.config.values["jax_platforms"]:
            jax.config.update("jax_platforms", "cpu")

    JAX_AVAILABLE = True
    FAILED_JAX_IMPORT = None
except ImportError as err:
    JAX_AVAILABLE = False
    FAILED_JAX_IMPORT = err

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
            "JAX is not available, using numpy to calculate einsum_aij_bji_to_ab."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    def prediction_einsum_aij_bji_to_ab_jax(
        given_operators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
        estimators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ) -> tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
        """Calculate the prediction of given operators by einsum_aij_bji_to_ab_jax.

        Args:
            given_operators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                The given operators.
            estimators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
                The estimators.

        Returns:
            tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
                A tuple containing:
                - A list of median values for each given operator.
                - A list of the corresponding median estimators for each given operator.
        """
        raise PostProcessingThirdPartyImportError(
            "JAX is not available, using numpy to calculate prediction_einsum_aij_bji_to_ab."
            + "error: "
            + str(FAILED_JAX_IMPORT)
        ) from FAILED_JAX_IMPORT

    def set_cpu_only():
        """Set JAX to use CPU only."""
        warnings.warn(
            "JAX is not available, nothing to set." + "error: " + str(FAILED_JAX_IMPORT),
            PostProcessingThirdPartyUnavailableWarning,
        )


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
        NDArray[np.complex128]: The Kronecker product of the :math:`\rho_{mki}`.
    """
    return ft.reduce(
        np.kron, [PRECOMPUTED_RHO_M_K_I[key] for key in key_list_of_precomputed]
    )  # type: ignore


def rho_mki_kronecker_product_numpy_2(
    key_list_of_precomputed: Iterable[int],
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    r"""Kronecker product for :math:`\rho_{mki}` by Numpy.

    Args:
        key_list_of_precomputed (Iterable[int]):
            The list of the keys of the precomputed :math:`\rho_{mki}`.

    Returns:
        NDArray[np.complex128]: The Kronecker product of the :math:`\rho_{mki}`.
    """
    return ft.reduce(
        np.kron, [PRECOMPUTED_RHO_M_K_I_2[key] for key in key_list_of_precomputed]
    )  # type: ignore


# single trace calculation
def single_trace_rho_by_trace_of_matmul(
    rho_m1_and_rho_m2: tuple[
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    ],
) -> np.complex128:
    """The single trace of Rho by trace of matmul.

    Args:
        rho_m1_and_rho_m2 (tuple[
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        ]):
            The tuple of rho_m1 and rho_m2.

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
        rho_m1_and_rho_m2 (tupletuple[
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    ]):
            The tuple of rho_m1 and rho_m2.

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
    np.complex128,
]:
    """Select the method to calculate the trace of Rho square.

    Args:
        method (str): The method to use for the calculation.

    Returns:
        Callable[[tuple[
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
            np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        ]], np.complex128]:
            The function to calculate the trace of Rho.
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


def prediction_einsum_aij_bji_to_ab_numpy(
    given_operators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    estimators: np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
) -> tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
    """Calculate the prediction of given operators by einsum_aij_bji_to_ab_numpy.

    Args:
        given_operators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
            The given operators.
        estimators (np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]):
            The estimators.

    Returns:
        tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]:
            A tuple containing:
            - A list of median values for each given operator.
            - A list of the corresponding median estimators for each given operator.
    """
    candidate_esitmators_foreach_given_operator = np.einsum(
        "aij,bji->ab", given_operators, estimators
    )
    median_foreach_given_operator = np.median(candidate_esitmators_foreach_given_operator, axis=1)
    median_location_given_operator = np.argmin(
        np.abs(
            candidate_esitmators_foreach_given_operator - median_foreach_given_operator[:, None]
        ),
        axis=1,
    )

    return list(median_foreach_given_operator), [
        candidate_esitmators_foreach_given_operator[i, j]
        for i, j in enumerate(median_location_given_operator)
    ]


def select_prediction_einsum_aij_bji_to_ab(
    method: AllTraceRhoMethod = DEFAULT_ALL_TRACE_RHO_METHOD,
) -> Callable[
    [
        np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ],
    tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]],
]:
    """Select the method to calculate the prediction of given operators.

    Args:
        method (AllTraceRhoMethod, optional):
            The method to use for the calculation. Defaults to DEFAULT_ALL_TRACE_RHO_METHOD
            It can be either "jax" or "numpy".

    Returns:
        Callable[[
            np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
            np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]
        ], tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]]:
            The function to calculate the prediction of given operators.
    """
    if method == "einsum_aij_bji_to_ab_jax":
        if JAX_AVAILABLE:
            return prediction_einsum_aij_bji_to_ab_jax
        warnings.warn(
            "JAX is not available, using numpy to calculate prediction.",
            PostProcessingThirdPartyUnavailableWarning,
        )
    if method != "einsum_aij_bji_to_ab_numpy":
        raise ValueError(f"Invalid backend: {method}")
    return prediction_einsum_aij_bji_to_ab_numpy
