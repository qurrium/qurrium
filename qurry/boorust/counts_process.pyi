"""Boorust - Counts Process (:mod:`qurry.boorust.counts_process`)"""

from typing import Sequence, Optional, Union, Literal

# pylint:disable=unused-argument
def single_counts_recount_rust(
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

def counts_list_recount_rust(
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

def shot_counts_selected_clreg_checker(
    shots: int,
    counts: list[dict[str, int]],
    selected_classical_registers: Optional[Sequence[int]] = None,
) -> tuple[int, list[int]]:
    """Check whether the selected classical registers are valid by Rust.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        selected_classical_registers (Optional[Sequence[int]], optional):
            The selected classical registers. Defaults to None.

    Returns:
        tuple[int, list[int]]:
            The size of the subsystem and the selected classical registers.
    """

def counts_list_vectorize_rust(
    counts_list: list[dict[str, int]],
) -> list[tuple[list[list[int]], list[int]]]:
    """Vectorized counts.

    Args:
        counts_list (list[dict[str, int]]):
            The list of counts measured from the single quantum circuit.

    Returns:
        list[tuple[list[list[int]], list[int]]]: The counts under the degree.
    """

def rho_m_flatten_counts_list_vectorize_rust(
    counts_list: list[dict[str, int]],
    random_unitary_um: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers_sorted: list[int],
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
