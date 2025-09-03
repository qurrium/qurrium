"""Post Processing - Randomized Measure - Entangled Entropy - Purity Cell 2
(:mod:`qurry.process.randomized_measure.entangled_entropy.purity_cell_2`)

This version introduces another way to process subsystems.

"""

import numpy as np

from ...utils import (
    ensemble_cell as ensemble_cell_py,
    single_counts_recount as single_counts_under_degree_py,
)


def purity_cell_2_py(
    idx: int,
    single_counts: dict[str, int],
    selected_classical_registers: list[int],
) -> tuple[int, np.float64, list[int]]:
    """Calculate the purity cell, one of overlap, of a subsystem by Python.

    Args:
        idx (int):
            Index of the cell (counts).
        single_counts (dict[str, int]):
            Counts measured from the single quantum circuit.
        selected_classical_registers (list[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[int, float, list[int]]:
            Index, one of overlap purity,
            The list of **the index of the selected classical registers**.
    """

    num_classical_register = len(list(single_counts.keys())[0])
    shots = sum(single_counts.values())

    selected_classical_registers_sorted = sorted(selected_classical_registers, reverse=True)
    subsystem_size = len(selected_classical_registers_sorted)
    single_counts_under_degree = single_counts_under_degree_py(
        single_counts, num_classical_register, selected_classical_registers_sorted
    )

    purity_cell_value = np.float64(0)
    for s_ai, s_ai_meas in single_counts_under_degree.items():
        for s_aj, s_aj_meas in single_counts_under_degree.items():
            purity_cell_value += ensemble_cell_py(
                s_ai, s_ai_meas, s_aj, s_aj_meas, subsystem_size, shots
            )

    return idx, purity_cell_value, selected_classical_registers_sorted
