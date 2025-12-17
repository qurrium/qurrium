"""Post Processing - Randomized Measure - Wavefunction Overlap - Echo Cell 2
(:mod:`qurry.process.randomized_measure.wavefunction_overlap.echo_cell_2`)

"""

import numpy as np

from ...utils import ensemble_cell as ensemble_cell_py, single_counts_recount_proto


def echo_cell_2_py(
    idx: int,
    first_single_counts: dict[str, int],
    second_single_counts: dict[str, int],
    selected_classical_registers: list[int],
) -> tuple[int, np.float64, list[int]]:
    """Calculate the echo cell, one of overlap, of a subsystem by Python.

    Args:
        idx (int):
            Index of the cell (counts).
        first_single_counts (dict[str, int]):
            Counts measured from the first quantum circuit.
        second_single_counts (dict[str, int]):
            Counts measured from the second quantum circuit.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[int, float, list[int]]:
            Index, one of overlap purity,
            The list of **the index of the selected classical registers**.
    """

    num_classical_register = len(list(first_single_counts.keys())[0])
    num_classical_register_02 = len(list(second_single_counts.keys())[0])
    assert num_classical_register == num_classical_register_02, (
        "The number of classical registers from the first and second counts are different. "
        + f"first: {num_classical_register}, second: {num_classical_register_02}"
    )

    shots = sum(first_single_counts.values())
    shots_02 = sum(second_single_counts.values())
    assert shots == shots_02, (
        "The shots from the first and second counts are different. "
        + f"first: {shots}, second: {shots_02}"
    )
    select_clregs_sort_rev = sorted(selected_classical_registers, reverse=True)
    subsystem_size = len(select_clregs_sort_rev)

    first_counts_under_degree = single_counts_recount_proto(
        first_single_counts, num_classical_register, select_clregs_sort_rev
    )
    second_counts_under_degree = single_counts_recount_proto(
        second_single_counts, num_classical_register, select_clregs_sort_rev
    )

    echo_cell_value = np.float64(0)
    for s_ai, s_ai_meas in first_counts_under_degree.items():
        for s_aj, s_aj_meas in second_counts_under_degree.items():
            echo_cell_value += ensemble_cell_py(
                s_ai, s_ai_meas, s_aj, s_aj_meas, subsystem_size, shots
            )

    return idx, echo_cell_value, select_clregs_sort_rev
