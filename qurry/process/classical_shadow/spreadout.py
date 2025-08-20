r"""Post Processing - Classical Shadow - Snapshots/Shots Spreadout
(:mod:`qurry.process.classical_shadow.spreadout`)

"""

from typing import Literal, Union, Iterable

from ..utils import check_invalid_counts
from ..classical_shadow import check_random_basis


def spreadout_counts(counts: list[dict[str, int]]) -> list[dict[str, int]]:
    """Spreadout the counts from multiple shots per snapshot to single shots counts.

    Args:
        counts (list[dict[str, int]]): The list of the counts.

    Returns:
        list[dict[str, int]]: The spreadout counts.
    """

    return [
        {bitstring: 1}
        for single_counts in counts
        for bitstring, counting in single_counts.items()
        for _ in range(counting)
    ]


def spreadout(
    shots: int,
    counts: list[dict[str, int]],
    random_basis: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Iterable[int],
) -> tuple[int, list[dict[str, int]], dict[int, dict[int, Union[Literal[0, 1, 2], int]]]]:
    """Spreadout the counts and random basis from multiple shots per snapshot
    to single shot per snapshot for classical shadow post-processing.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.
        selected_classical_registers (Iterable[int]):
            The list of **the index of the selected_classical_registers**.

    Returns:
        tuple[int, list[dict[str, int]], dict[int, dict[int, Union[Literal[0, 1, 2], int]]]]:
            The spreadout snapshotss shots, counts, and random basis.
    """

    if len(counts) != len(random_basis):
        raise ValueError("The length of counts and random_basis must be the same.")
    check_invalid_counts(shots, counts)
    check_random_basis(random_basis, list(selected_classical_registers))

    new_snapshots = len(random_basis) * shots
    new_random_basis = {
        (i * shots + j): single_random_basis
        for i, single_random_basis in random_basis.items()
        for j in range(shots)
    }
    new_counts = spreadout_counts(counts)

    return new_snapshots, new_counts, new_random_basis
