"""Post Processing - Utils - Counts Process (:mod:`qurry.process.utils.ccounts_process`)"""

import warnings
from typing import Union, Optional, Literal, Iterable

from ..availability import availablility, PostProcessingBackendLabel
from ..exceptions import PostProcessingRustImportError, PostProcessingRustUnavailableWarning

try:
    from ...boorust import counts_process  # type: ignore

    single_counts_recount_rust_source = counts_process.single_counts_recount_rust
    counts_list_recount_rust_source = counts_process.counts_list_recount_rust
    shot_counts_selected_clreg_checker_source = counts_process.shot_counts_selected_clreg_checker
    counts_list_vectorize_rust_source = counts_process.counts_list_vectorize_rust
    rho_m_flatten_counts_list_vectorize_rust_source = (
        counts_process.rho_m_flatten_counts_list_vectorize_rust
    )

    RUST_AVAILABLE = True
    FAILED_RUST_IMPORT = None
except ImportError as err:
    RUST_AVAILABLE = False
    FAILED_RUST_IMPORT = err

    def single_counts_recount_rust_source(*args, **kwargs):
        """Dummy function for counts_under_degree_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate counts under degree."
        ) from FAILED_RUST_IMPORT

    def counts_list_recount_rust_source(*args, **kwargs):
        """Dummy function for counts_list_under_degree_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate counts list under degree."
        ) from FAILED_RUST_IMPORT

    def shot_counts_selected_clreg_checker_source(*args, **kwargs):
        """Dummy function for shot_counts_selected_clreg_checker."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate shot counts selected clreg checker."
        ) from FAILED_RUST_IMPORT

    def counts_list_vectorize_rust_source(*args, **kwargs):
        """Dummy function for counts_list_vectorized_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate counts list vectorized."
        ) from FAILED_RUST_IMPORT

    def rho_m_flatten_counts_list_vectorize_rust_source(*args, **kwargs):
        """Dummy function for rho_m_flatten_counts_list_vectorized_rust."""
        raise PostProcessingRustImportError(
            "Rust is not available, using python to calculate rho_m_flatten counts list vectorized."
        ) from FAILED_RUST_IMPORT


BACKEND_AVAILABLE = availablility(
    "utils.counts_process",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)
DEFAULT_PROCESS_BACKEND = "Rust" if RUST_AVAILABLE else "Python"


def check_invalid_counts(shots: int, counts: list[dict[str, int]]):
    """Check whether the counts are valid.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.

    Raises:
        ValueError: If the counts are invalid, which some of them mismatch shots number.
    """
    invalid_counts = list(
        filter(lambda i_and_c: sum(i_and_c[1].values()) != shots, enumerate(counts))
    )
    if len(invalid_counts) > 0:
        raise ValueError(
            "The counts must be equal to the number of shots, "
            + f"but following counts are invalid, index: {invalid_counts}"
        )


def single_counts_recount(
    single_counts: dict[str, int],
    num_classical_register: int,
    selected_classical_registers_sorted: list[int],
) -> dict[str, int]:
    """Calculate the counts under the degree.

    Args:
        single_counts (dict[str, int]):
            Counts measured from the single quantum circuit.
        num_classical_register (int):
            The number of classical registers.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        dict[str, int]: The counts under the degree.
    """

    single_counts_recounted = {}
    for bitstring_all, num_counts_all in single_counts.items():
        bitstring = "".join(
            bitstring_all[num_classical_register - q_i - 1]
            for q_i in selected_classical_registers_sorted
        )
        if bitstring in single_counts_recounted:
            single_counts_recounted[bitstring] += num_counts_all
        else:
            single_counts_recounted[bitstring] = num_counts_all

    return single_counts_recounted


def counts_list_recount(
    counts_list: list[dict[str, int]],
    num_classical_register: int,
    selected_classical_registers_sorted: list[int],
) -> list[dict[str, int]]:
    """Calculate the counts under the degree.

    Args:
        counts_list (list[dict[str, int]]):
            The list of counts measured from the single quantum circuit.
        num_classical_register (int):
            The number of classical registers.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        list[dict[str, int]]: The counts under the degree.
    """
    return [
        single_counts_recount(
            single_counts, num_classical_register, selected_classical_registers_sorted
        )
        for single_counts in counts_list
    ]


def single_counts_recount_pyrust(
    single_counts: dict[str, int],
    num_classical_register: int,
    selected_classical_registers_sorted: list[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> dict[str, int]:
    """Calculate the counts under the degree.

    Args:
        single_counts (dict[str, int]):
            Counts measured from the single quantum circuit.
        num_classical_register (int):
            The number of classical registers.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to "Rust".

    Returns:
        dict[str, int]: The counts under the degree.
    """

    if backend == "Rust":
        if RUST_AVAILABLE:
            return single_counts_recount_rust_source(
                single_counts, num_classical_register, selected_classical_registers_sorted
            )
        warnings.warn(
            "Rust is not available, using python to calculate counts under degree."
            + f" Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
        backend = "Python"
    if backend != "Python":
        warnings.warn(
            f"Invalid backend '{backend}', using Python to calculate counts under degree. "
            + "The backend should be 'Python' or 'Rust'.",
            PostProcessingRustUnavailableWarning,
        )
    return single_counts_recount(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )


def counts_list_recount_pyrust(
    counts_list: list[dict[str, int]],
    num_classical_register: int,
    selected_classical_registers_sorted: list[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> list[dict[str, int]]:
    """Calculate the counts under the degree.

    Args:
        counts_list (list[dict[str, int]]):
            The list of counts measured from the single quantum circuit.
        num_classical_register (int):
            The number of classical registers.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to "Rust".

    Returns:
        list[dict[str, int]]: The counts under the degree.
    """
    if backend == "Rust":
        if RUST_AVAILABLE:
            return counts_list_recount_rust_source(
                counts_list, num_classical_register, selected_classical_registers_sorted
            )
        warnings.warn(
            "Rust is not available, using python to calculate counts under degree."
            + f" Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
        backend = "Python"
    if backend != "Python":
        warnings.warn(
            f"Invalid backend '{backend}', using Python to calculate counts under degree. "
            + "The backend should be 'Python' or 'Rust'.",
            PostProcessingRustUnavailableWarning,
        )
    return counts_list_recount(
        counts_list, num_classical_register, selected_classical_registers_sorted
    )


def selected_clregs_to_optlist(
    selected_classical_registers: Optional[Iterable[int]] = None,
) -> Optional[list[int]]:
    """Convert selected classical registers to a list.
    This usually uses for Rust binding
    which can not handle :class:`~typing.Iterable` but only :class:`list`.

    Args:
        selected_classical_registers (Optional[Iterable[int]], optional):
            The selected classical registers.

    Returns:
        Optional[list[int]]: The list of selected classical registers or None.
    """

    if isinstance(selected_classical_registers, Iterable):
        selected_classical_registers = list(selected_classical_registers)
    elif selected_classical_registers is not None:
        raise TypeError(
            "selected_classical_registers must be an Iterable or None"
            + f", but got {type(selected_classical_registers)}"
        )

    return selected_classical_registers


def shot_counts_selected_clreg_checker(
    shots: int,
    counts: list[dict[str, int]],
    selected_classical_registers: Optional[Iterable[int]] = None,
) -> tuple[int, list[int]]:
    """Check whether the selected classical registers are valid.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The selected classical registers. Defaults to None.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to "Rust".

    Returns:
        tuple[int, list[int]]:
            The size of the subsystem and the selected classical registers.
    """

    check_invalid_counts(shots, counts)

    # Determine subsystem size
    measured_system_size = len(list(counts[0].keys())[0])

    if selected_classical_registers is None:
        selected_classical_registers = list(range(measured_system_size))
    elif not isinstance(selected_classical_registers, Iterable):
        raise ValueError(
            "selected_classical_registers should be Iterable, "
            + f"but get {type(selected_classical_registers)}"
        )
    else:
        selected_classical_registers = list(selected_classical_registers)
    assert all(
        0 <= q_i < measured_system_size for q_i in selected_classical_registers
    ), f"Invalid selected classical registers: {selected_classical_registers}"

    return measured_system_size, selected_classical_registers


def shot_counts_selected_clreg_checker_rust(
    shots: int,
    counts: list[dict[str, int]],
    selected_classical_registers: Optional[Iterable[int]] = None,
) -> tuple[int, list[int]]:
    """Check whether the selected classical registers are valid by Rust.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The selected classical registers. Defaults to None.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to "Rust".

    Returns:
        tuple[int, list[int]]:
            The size of the subsystem and the selected classical registers.
    """
    selected_classical_registers = selected_clregs_to_optlist(selected_classical_registers)
    return shot_counts_selected_clreg_checker_source(shots, counts, selected_classical_registers)


def shot_counts_selected_clreg_checker_pyrust(
    shots: int,
    counts: list[dict[str, int]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> tuple[int, list[int]]:
    """Check whether the selected classical registers are valid.
    This function wraps the implementation of Python and Rust.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The selected classical registers. Defaults to None.
        backend (PostProcessingBackendLabel, optional):
            Backend for the process. Defaults to "Rust".

    Returns:
        tuple[int, list[int]]:
            The size of the subsystem and the selected classical registers.
    """
    if backend == "Rust":
        if RUST_AVAILABLE:
            return shot_counts_selected_clreg_checker_rust(
                shots, counts, selected_classical_registers
            )
        warnings.warn(
            "Rust is not available, using python to calculate shot counts selected clreg checker."
            + f" Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )

    return shot_counts_selected_clreg_checker(shots, counts, selected_classical_registers)


def counts_list_vectorize_pyrust(
    counts_list: list[dict[str, int]],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> list[tuple[list[list[int]], list[int]]]:
    """Vectorized counts.

    Args:
        counts_list (list[dict[str, int]]):
            The list of counts measured from the single quantum circuit.

    Returns:
        list[tuple[list[list[int]], list[int]]]: The counts under the degree.
    """
    if backend == "Rust":
        if RUST_AVAILABLE:
            return counts_list_vectorize_rust_source(counts_list)
        warnings.warn(
            "Rust is not available, using python to calculate counts list vectorized."
            + f" Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
        backend = "Python"

    vectorized_counts = []
    for single_counts in counts_list:
        keys_int_array: list[list[int]] = [list(map(int, k)) for k in single_counts.keys()]
        values_int_array: list[int] = list(single_counts.values())

        vectorized_counts.append((keys_int_array, values_int_array))
    return vectorized_counts


def rho_m_flatten_counts_list_vectorize_pyrust(
    counts_list: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers_sorted: list[int],
    backend: PostProcessingBackendLabel = DEFAULT_PROCESS_BACKEND,
) -> list[tuple[list[list[int]], list[int]]]:
    """Dedicated function for rho_m_flatten counts list vectorized.

    Args:
        counts_list (list[dict[str, int]]):
            The list of counts measured from the single quantum circuit.
        random_unitary_um (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The shadow direction of the unitary operators.
        selected_classical_registers_sorted (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        list[tuple[list[list[int]], list[int]]]: The counts under the degree.
    """
    if backend == "Rust":
        if RUST_AVAILABLE:
            return rho_m_flatten_counts_list_vectorize_rust_source(
                counts_list, random_unitary_um, selected_classical_registers_sorted
            )
        warnings.warn(
            "Rust is not available, using python to calculate rho_m_flatten counts list vectorized."
            + f" Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
        backend = "Python"

    rho_m_flatten_vectorized_counts = []
    for um_idx, single_counts in enumerate(counts_list):
        keys_int_array: list[list[int]] = [
            [
                int(c) + 10 * random_unitary_um[um_idx][selected_classical_registers_sorted[q_idx]]
                for q_idx, c in enumerate(bit_string)
            ]
            for bit_string in single_counts.keys()
        ]
        values_int_array: list[int] = list(single_counts.values())

        rho_m_flatten_vectorized_counts.append((keys_int_array, values_int_array))
    return rho_m_flatten_vectorized_counts
