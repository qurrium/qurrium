"""Post Processing - String Operator - String Operator Core
(:mod:`qurry.process.string_operator.strop_core`)

"""

import warnings
from typing import Union, Callable, Literal
import numpy as np

from ..availability import (
    availablility,
    default_postprocessing_backend,
    PostProcessingBackendLabel,
)
from ..exceptions import (
    # PostProcessingRustImportError,
    PostProcessingRustUnavailableWarning,
)


# try:
#     from ...boorust import magsq  # type: ignore

#     string_operator_core_rust_source = magsq.string_operator_core_rust

#     RUST_AVAILABLE = True
#     FAILED_RUST_IMPORT = None
# except ImportError as err:
#     RUST_AVAILABLE = False
#     FAILED_RUST_IMPORT = err

#     def string_operator_core_rust_source(*args, **kwargs):
#         """Dummy function for string_operator_core_rust."""
#         raise PostProcessingRustImportError(
#             "Rust is not available, using python to calculate string operator."
#         ) from FAILED_RUST_IMPORT


BACKEND_AVAILABLE = availablility(
    "string_operator.strop_core",
    [
        # ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)
DEFAULT_PROCESS_BACKEND = default_postprocessing_backend(
    # RUST_AVAILABLE, False
    False,
    False,
)


# def string_operator_core_allrust(
#     shots: int,
#     counts: list[dict[str, int]],
#     num_qubits: int,
# ) -> tuple[Union[float, np.float64], dict[int, Union[float, np.float64]], int, float]:
#     """The core function of string operator by Rust."""
#     return string_operator_core_rust_source(counts, shots, num_qubits)


add_or_reducer: Callable[[str], Literal[1, -1]] = lambda bitstring: (
    1 if sum(int(bit) for bit in bitstring) % 2 == 0 else -1
)
"""The add or reduce function.
If the sum of the bitstring is even, return 1.
If the sum of the bitstring is odd, return -1.

Args:
    bitstring (str): The bitstring.
Returns:
    Literal[1, -1]: 1 or -1.
"""


def string_operator_core(
    counts: list[dict[str, int]],
    shots: int,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> Union[float, np.float64]:
    """The core function of magnet square by Python and Rust.

    Args:
        counts (list[dict[str, int]]):
            Counts of the experiment on quantum machine.
        shots (int):
            Shots of the experiment on quantum machine.
        backend (PostProcessingBackendLabel, optional):
            Post Processing backend. Defaults to DEFAULT_PROCESS_BACKEND.

    Returns:
        Union[float, np.float64]: String operator value.
    """
    if len(counts) != 1:
        raise ValueError(f"counts should be a list of counts with length 1, but got {len(counts)}")

    if backend == "Rust":
        warnings.warn(
            PostProcessingRustUnavailableWarning(
                "Rust is not ready, using Python to calculate magnetic square."
            )
        )
        # return string_operator_core_allrust(counts, shots, num_qubits)

    only_counts = counts[0]

    sample_shots = sum(only_counts.values())
    assert sample_shots == shots, f"shots {shots} does not match sample_shots {sample_shots}"

    order_per_bitstring_without_div_by_shots = {
        s: add_or_reducer(s) * m for s, m in only_counts.items()
    }
    order = sum(order_per_bitstring_without_div_by_shots.values()) / sample_shots

    return order
