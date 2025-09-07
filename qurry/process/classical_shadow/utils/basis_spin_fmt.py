"""Post Processing - Classical Shadow - Utilities - Basis/Spin Format
(:mod:`qurry.process.classical_shadow.utils.basis_spin_fmt`)

This module provides utility functions to convert the format of basis and spin outcomes
which use in `Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_ ,
between different representations used in counts and random basis.

And export the format to a text file that can be read by
`Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_ .

- Single shots basis-spin Output format (for PQP)

.. code-block:: text

    [system size]
    [X/Y/Z for qubit 1] [-1/1 for qubit 1] [X/Y/Z for qubit 2] [-1/1 for qubit 2] ...
    [X/Y/Z for qubit 1] [-1/1 for qubit 1] [X/Y/Z for qubit 2] [-1/1 for qubit 2] ...

"""

from typing import Literal, Union

from .spreadout import spreadout


def validate_counts_and_basis(
    idx: int,
    single_counts: dict[str, int],
    single_random_basis: dict[int, int],
) -> str:
    """Validate the single counts and random basis, then return bitstring.

    Args:
        idx (int): The index of the current counts and random basis
        single_counts (dict[str, int]): The counts of single-shot results.
        single_random_basis (dict[int, int]): Mapping of qubit indices to random basis.

    Returns:
        str: The validated bitstring.
    """
    bitstring = next(iter(single_counts.keys()))
    if len(bitstring) != len(single_random_basis):
        raise ValueError(
            "The length of the bitstring must match the number of qubits in single_random_basis. "
            + f"Bitstring and its length: '{bitstring}', {len(bitstring)}. "
            + f"Random basis and its length: '{single_random_basis}', {len(single_random_basis)}. "
            + f"Index: {idx}."
        )
    return bitstring


def combine_counts_and_basis(
    idx: int,
    single_counts: dict[str, int],
    single_random_basis: dict[int, int],
) -> tuple[list[int], list[int]]:
    """Single counts processing for PQP.

    Args:
        idx (int): The index of the current counts and random basis
        single_counts (dict[str, int]): The counts of single-shot results.
        single_random_basis (dict[int, int]): Mapping of qubit indices to random basis.

    Returns:
        tuple[list[int], list[int]]:
            A tuple containing pauli basis and spin outcomes.
    """

    return list(single_random_basis.values()), [
        int(bits) if bits == "1" else -1
        for bits in reversed(validate_counts_and_basis(idx, single_counts, single_random_basis))
    ]


def multi_counts_to_basis_spin(
    shots: int,
    counts: list[dict[str, int]],
    random_basis: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Convert a Qurrium experiment to Predicting Quantum Properties result.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.

    Returns:
        tuple[list[list[int]], list[list[int]]]:
            A tuple containing a list of pauli basis and a list of spin outcomes
    """
    _shots, singleshot_counts, singleshot_random_basis = spreadout(shots, counts, random_basis)

    results = [
        combine_counts_and_basis(idx, single_counts, single_random_basis)
        for (idx, single_random_basis), single_counts in zip(
            singleshot_random_basis.items(), singleshot_counts
        )
    ]

    pauli_basis, spin_outcome = map(list, zip(*results)) if results else ([], [])

    return pauli_basis, spin_outcome
