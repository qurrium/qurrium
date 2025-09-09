"""Post Processing - Classical Shadow - Utilities - Basis/Spin Format
(:mod:`qurry.process.classical_shadow.utils.basis_spin_fmt`)

This module provides utility functions to convert the format of basis and spin outcomes
which use in `Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_ ,
between different representations used in counts and random basis.

And export the format to a text file that can be read by
`Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_ .

"""

from typing import Literal, Union, Sequence
from pathlib import Path


def multi_counts_to_basis_spin(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[Union[Literal[0, 1, 2], int]]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert counts to single shots basis-spin format.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            A tuple containing a list of pauli basis and a list of spin outcomes
    """
    pauli_basis, spin_outcome = [], []
    for idx, (single_counts, single_random_basis) in enumerate(zip(counts, random_basis_array)):
        partial_pauli_basis = [single_random_basis for _ in range(shots)]
        partial_spin_outcome = []

        for bitstring, count in single_counts.items():
            spin_values = [1 if bits == "1" else -1 for bits in reversed(bitstring)]
            partial_spin_outcome.extend([spin_values] * count)

        if len(partial_spin_outcome) != shots:
            raise ValueError(
                "The single counts should only contain one bitstring with count equal to shots. "
                + f"Counts: {single_counts}. Index: {idx}."
            )
        pauli_basis.extend(partial_pauli_basis)
        spin_outcome.append(partial_spin_outcome)

    return pauli_basis, spin_outcome


def measurements_export(
    pauli_basis: Sequence[Sequence[int]],
    spin_outcome: Sequence[Sequence[int]],
    system_size: int,
    filename: Union[str, Path],
) -> Union[str, Path]:
    """Export the measurement data.

    Args:
        pauli_basis (Sequence[Sequence[int]]):
            The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (Sequence[Sequence[int]]):
            The list of spin outcomes. (1, -1)
        system_size (int):
            The size of the quantum system, which is the number of qubits.
        filename (Union[str, Path]):
            The path to the output file.

    Returns:
        The path to the output file.
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{system_size}\n")
        for single_pauli, single_spin in zip(pauli_basis, spin_outcome):
            f.writelines(
                " ".join(
                    (f"{chr(88 + pauli)} {spin}" for pauli, spin in zip(single_pauli, single_spin))
                )
            )

    return filename
