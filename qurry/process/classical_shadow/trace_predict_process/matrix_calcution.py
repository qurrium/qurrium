"""Post Processing - Classical Shadow - Trace-Preidction Process - Matrix Calculation
(:mod:`qurry.process.classical_shadow.trace_predict_process.matrix_calcution`)

The matrix calculattion for predicting quantum properties.

"""

from typing import Callable, Union
import warnings
import numpy as np

from ...utils import BaseMethodEnum
from ...availability import availablility
from ...exceptions import (
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
    [("Numpy", True, None), ("jax", JAX_AVAILABLE, FAILED_JAX_IMPORT)],
)
"""The availability of backends for classical shadow matrix calculation."""


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


class SingleTraceMethod(BaseMethodEnum):
    """The method to calculate the trace of single Rho square.

    - "trace_of_matmul": Use `np.trace(np.matmul(rho_m1, rho_m2))` to calculate the trace.
    - "einsum_ij_ji": Use `np.einsum("ij,ji", rho_m1, rho_m2)` to calculate the trace.
    """

    TRACE_OF_MATMUL = "trace_of_matmul"
    """Use `np.trace(np.matmul(rho_m1, rho_m2))` to calculate the trace."""

    EINSUM_IJ_JI = "einsum_ij_ji"
    """Use `np.einsum("ij,ji", rho_m1, rho_m2)` to calculate the trace."""

    @classmethod
    def get_default(cls) -> "SingleTraceMethod":
        """Get the default method.

        Returns:
            The default method.
        """
        return cls.EINSUM_IJ_JI


SingleTraceMethodType = Union[SingleTraceMethod, str]
"""The method to use for the trace calculation with matrix multiplication.
- "trace_of_matmul":
    Use `np.trace(np.matmul(rho_m1, rho_m2))` to calculate the trace.
- "einsum_ij_ji":
    Use `np.einsum("ij,ji", rho_m1, rho_m2)` to calculate the trace.
"""

DEFAULT_SINGLE_TRACE_METHOD: SingleTraceMethod = SingleTraceMethod.get_default()
"""The default method for the trace calculation with matrix multiplication."""


def select_single_trace_rho_method(
    method: SingleTraceMethodType = DEFAULT_SINGLE_TRACE_METHOD,
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
        method (SingleTraceMethodType): The method to use for the calculation.

    Returns:
        The function to calculate the trace of Rho.
    """

    if isinstance(method, str):
        method = SingleTraceMethod.from_string(method)
    if SingleTraceMethod.EINSUM_IJ_JI == method:
        return single_trace_rho_by_einsum_ij_ji
    if method == SingleTraceMethod.TRACE_OF_MATMUL:
        return single_trace_rho_by_trace_of_matmul

    raise SingleTraceMethod.value_error()


class ListTraceMethod(BaseMethodEnum):
    """The method to calculate the all trace of Rho square.

    - "einsum_aij_bji_to_ab_numpy": Use\
    `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho if JAX is not available.
    - "einsum_aij_bji_to_ab_jax": Use\
    `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho.
    """

    EINSUM_AIJ_BJI_TO_AB_NUMPY = "einsum_aij_bji_to_ab_numpy"
    """Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""

    EINSUM_AIJ_BJI_TO_AB_JAX = "einsum_aij_bji_to_ab_jax"
    """Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace."""

    @classmethod
    def get_default(cls) -> "ListTraceMethod":
        """Get the default method.

        Returns:
            The default method.
        """
        return cls.EINSUM_AIJ_BJI_TO_AB_JAX if JAX_AVAILABLE else cls.EINSUM_AIJ_BJI_TO_AB_NUMPY

    def handle_jax_unavailability(self) -> "ListTraceMethod":
        """Handle JAX unavailability by falling back to numpy method if necessary.

        Returns:
            ListTraceMethod: The original method if JAX is available or not needed,
            otherwise the numpy method.
        """
        if self == self.EINSUM_AIJ_BJI_TO_AB_JAX and not JAX_AVAILABLE:
            warnings.warn(
                "JAX is not available, using numpy to calculate all trace.",
                PostProcessingThirdPartyUnavailableWarning,
            )
            return self.EINSUM_AIJ_BJI_TO_AB_NUMPY
        return self


ListTraceMethodType = Union[ListTraceMethod, str]
"""The method to calculate the all trace of Rho square.

- "einsum_aij_bji_to_ab_numpy":
    Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho
    if JAX is not available.
- "einsum_aij_bji_to_ab_jax":
    Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
    This is the fastest implementation to calculate the trace of Rho.
"""

DEFAULT_LIST_TRACE_METHOD: ListTraceMethod = ListTraceMethod.get_default()
"""The default method for the trace calculation with matrix multiplication."""


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


def select_all_trace_rho_by_einsum_aij_bji_to_ab(
    method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
) -> Callable[
    [np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]],
    np.complex128,
]:
    """Select the method to calculate the trace of Rho square.

    Args:
        method (ListTraceMethodType, optional):
            The method to use for the calculation.

            - "einsum_aij_bji_to_ab_numpy":
                Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho
                if JAX is not available.
            - "einsum_aij_bji_to_ab_jax":
                Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho.

            Defaults to DEFAULT_LIST_TRACE_METHOD.

    Returns:
        Callable[[np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]], np.complex128]:
            The function to calculate the trace of Rho.
    """
    if isinstance(method, str):
        method = ListTraceMethod.from_string(method)
    method = method.handle_jax_unavailability()
    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX:
        return all_trace_rho_by_einsum_aij_bji_to_ab_jax
    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY:
        return all_trace_rho_by_einsum_aij_bji_to_ab_numpy

    raise ListTraceMethod.value_error()


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
    method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
) -> Callable[
    [
        np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
        np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
    ],
    tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]],
]:
    """Select the method to calculate the prediction of given operators.

    Args:
        method (ListTraceMethodType, optional):
            The method to use for the calculation.

            - "einsum_aij_bji_to_ab_numpy":
                Use `np.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho
                if JAX is not available.
            - "einsum_aij_bji_to_ab_jax":
                Use `jnp.einsum("aij,bji->ab", rho_m_list, rho_m_list)` to calculate the trace.
                This is the fastest implementation to calculate the trace of Rho.

            Defaults to DEFAULT_LIST_TRACE_METHOD.

    Returns:
        Callable[[
            np.ndarray[tuple[int, int, int], np.dtype[np.complex128]],
            np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]
        ], tuple[list[np.complex128], list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]]:
            The function to calculate the prediction of given operators.
    """
    if isinstance(method, str):
        method = ListTraceMethod.from_string(method)
    method = method.handle_jax_unavailability()
    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX:
        return prediction_einsum_aij_bji_to_ab_jax
    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY:
        return prediction_einsum_aij_bji_to_ab_numpy

    raise ListTraceMethod.value_error()
