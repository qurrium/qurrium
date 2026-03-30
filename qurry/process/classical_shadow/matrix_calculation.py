"""Post Processing - Classical Shadow - Trace-Prediction Process - Matrix Calculation
(:mod:`qurry.process.classical_shadow.trace_predict_process.matrix_calculation`)

The matrix calculation for predicting quantum properties.

"""

from collections.abc import Callable
import warnings
import numpy as np
import numpy.typing as npt

from ..utils import BaseMethodEnum
from ..availability import availablility
from ..exceptions import PostProcessingThirdPartyUnavailableWarning


# pylint: disable=import-outside-toplevel
def is_jax_available():
    """Check if JAX is available.

    Returns:
        tuple[bool, ImportError | None]: A tuple containing a boolean indicating
        whether JAX is available and the ImportError if it is not.
    """
    try:
        # pylint: disable=unused-import
        import jax  # noqa: F401
        # pylint: enable=unused-import

        return True, None
    except ImportError as e:
        return False, e


JAX_AVAILABLE, FAILED_JAX_IMPORT = is_jax_available()


def set_jax_platform_cpu_only():
    """Set JAX to use CPU only. We just want to use JAX for speed up on CPU.
    And we don't want to handle GPU/TPU issues in Qurrium."""
    if not JAX_AVAILABLE:
        warnings.warn(
            "JAX is not available, nothing to set. error: " + str(FAILED_JAX_IMPORT),
            PostProcessingThirdPartyUnavailableWarning,
        )
        return

    import jax

    jax.config.update("jax_platforms", "cpu")


def set_jax_enable_x64(enable: bool = True):
    """Set JAX to enable or disable 64-bit precision.

    This is required to handle the complex128 dtype in JAX.
    Or the result of JAX will be not same as Numpy.

    Args:
        enable (bool, optional): Whether to enable 64-bit precision. Defaults to True.
    """
    if not JAX_AVAILABLE:
        warnings.warn(
            "JAX is not available, nothing to set. error: " + str(FAILED_JAX_IMPORT),
            PostProcessingThirdPartyUnavailableWarning,
        )
        return

    import jax

    jax.config.update("jax_enable_x64", enable)


def check_jax_enabled_x64():
    """Check if JAX is enabled for 64-bit precision."""
    if not JAX_AVAILABLE:
        return

    import jax

    if not jax.config.values["jax_enable_x64"]:
        warnings.warn(
            "JAX is not set to use 64-bit precision, but it should be setup by Qurrium "
            + "Since we rely on 64-bit precision to confirm "
            + "that it made same result with Numpy. "
            + "You can set it by `jax.config.update('jax_enable_x64', True)`. "
            + "Or you can set it in your environment by `export JAX_ENABLE_X64=True`. ",
            RuntimeWarning,
        )


BACKEND_AVAILABLE = availablility(
    "classical_shadow.matrix_calculation",
    [("Numpy", True, None), ("JAX", JAX_AVAILABLE, FAILED_JAX_IMPORT)],
)
"""The availability of backends for classical shadow matrix calculation."""


# single trace calculation
def single_trace_rho_by_trace_of_matmul(
    rho_m1_and_rho_m2: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
) -> np.complex128:
    """The single trace of Rho by trace of matmul.

    Args:
        rho_m1_and_rho_m2 (tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]):
            The tuple of rho_m1 and rho_m2.
            It should be two 2-dimensional arrays for matrix multiplication.

    Returns:
        np.complex128: The trace of Rho.
    """
    rho_m1, rho_m2 = rho_m1_and_rho_m2
    return np.trace((rho_m1 @ rho_m2)) + np.trace((rho_m2 @ rho_m1))


def single_trace_rho_by_einsum_ij_ji(
    rho_m1_and_rho_m2: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
) -> np.complex128:
    """The single trace of Rho by einsum_ij_ji by Numpy.

    Args:
        rho_m1_and_rho_m2 (tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]):
            The tuple of rho_m1 and rho_m2.
            It should be two 2-dimensional arrays for matrix multiplication.

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


SingleTraceMethodType = SingleTraceMethod | str
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
) -> Callable[[tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]], np.complex128]:
    """Select the method for the single trace of Rho.

    Args:
        method (SingleTraceMethodType, optional):
            The method to use for the calculation. Defaults to DEFAULT_SINGLE_TRACE_METHOD.

    Returns:
        The function to calculate the single trace of Rho.
    """

    if isinstance(method, str):
        method = SingleTraceMethod.from_string(method)
    if method == SingleTraceMethod.EINSUM_IJ_JI:
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


ListTraceMethodType = ListTraceMethod | str
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
def all_trace_rho_by_einsum_aij_bji_to_ab(
    rho_m_array: npt.NDArray[np.complex128],
    method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
) -> np.complex128:
    """The trace of Rho by einsum_aij_bji_to_ab.
    This is the fastest implementation to calculate the trace of Rho.

    Args:
        rho_m_array (npt.NDArray[np.complex128]):
            The Rho M array.
            It should be a 3-dimensional array for a list of operators.
        method (ListTraceMethodType, optional):
            The method to use for the calculation.
            Defaults to DEFAULT_LIST_TRACE_METHOD.

    Returns:
        np.complex128: The trace of Rho.
    """
    if rho_m_array.ndim != 3:
        raise ValueError(
            f"rho_m_array must be a 3-dimensional array. Got {rho_m_array.ndim} dimensions."
        )
    if isinstance(method, str):
        method = ListTraceMethod.from_string(method)
    method = method.handle_jax_unavailability()

    len_rho_m_array = len(rho_m_array)

    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY:
        trace_matrix = np.einsum("aij,bji -> ab", rho_m_array, rho_m_array)
        mask = np.ones(trace_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)

        sum_off_diagonal = trace_matrix[mask].sum()
        return sum_off_diagonal / (len_rho_m_array * (len_rho_m_array - 1))

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)

    trace_matrix = jnp.einsum("aij,bji -> ab", rho_m_array, rho_m_array)
    mask = np.ones(trace_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)

    sum_off_diagonal = trace_matrix[mask].sum()
    return np.complex128(sum_off_diagonal / (len_rho_m_array * (len_rho_m_array - 1)))


def prediction_einsum_aij_bji_to_ab(
    given_operators: npt.NDArray[np.complex128],
    estimators: npt.NDArray[np.complex128],
    method: ListTraceMethodType = DEFAULT_LIST_TRACE_METHOD,
) -> tuple[list[np.complex128], list[npt.NDArray[np.complex128]]]:
    """Calculate the prediction of given operators by einsum_aij_bji_to_ab_numpy.

    Args:
        given_operators (npt.NDArray[np.complex128]):
            The given operators.
            It should be a 3-dimensional array for a list of operators.
        estimators (npt.NDArray[np.complex128]):
            The estimators.
            It should be a 3-dimensional array for a list of operators.
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
        A tuple containing:
            - A list of median values for each given operator.
            - A list of the corresponding median estimators for each given operator.
    """
    if given_operators.ndim != 3 or estimators.ndim != 3:
        raise ValueError(
            "given_operators and estimators must be 3-dimensional arrays."
            f"Got {given_operators.ndim} and {estimators.ndim} dimensions respectively."
        )

    if isinstance(method, str):
        method = ListTraceMethod.from_string(method)
    method = method.handle_jax_unavailability()

    if method == ListTraceMethod.EINSUM_AIJ_BJI_TO_AB_NUMPY:
        candidate_esitmators_foreach_given_operator = np.einsum(
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
            candidate_esitmators_foreach_given_operator[i, j]
            for i, j in enumerate(median_location_given_operator)
        ]

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)

    candidate_esitmators_foreach_given_operator = jnp.einsum(
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
        np.array(candidate_esitmators_foreach_given_operator[i, j], dtype=np.complex128)
        for i, j in enumerate(median_location_given_operator)
    ]
