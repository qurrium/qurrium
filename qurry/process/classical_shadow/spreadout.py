r"""Post Processing - Classical Shadow - Snapshots/Shots Spreadout
(:mod:`qurry.process.classical_shadow.spreadout`)

"""

from typing import Literal, Union


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


def spreadout_random_basis(
    shots: int, random_basis: dict[int, dict[int, Union[Literal[0, 1, 2], int]]]
) -> dict[int, dict[int, Union[Literal[0, 1, 2], int]]]:
    """Spreadout the random basis from multiple shots per snapshot to single shots counts.

    Args:
        random_basis (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The random basis to be spreadout.

    Returns:
        dict[int, dict[int, Union[Literal[0, 1, 2], int]]]: The spreadout random basis.
    """

    return {
        (i * shots + j): single_random_basis
        for i, single_random_basis in random_basis.items()
        for j in range(shots)
    }


def spreadout(
    shots: int,
    counts: list[dict[str, int]],
    random_basis: dict[int, dict[int, Union[Literal[0, 1, 2], int]]],
) -> tuple[list[dict[str, int]], dict[int, dict[int, Union[Literal[0, 1, 2], int]]]]:
    """Spreadout the counts and random basis from multiple shots per snapshot
    to single shot per snapshot for classical shadow post-processing.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis (dict[int, dict[int, Union[Literal[0, 1, 2], int]]]):
            The random basis for classical shadow.

    Returns:
        tuple[list[dict[str, int]], dict[int, dict[int, Union[Literal[0, 1, 2], int]]]]:
            The spreadout counts and random basis.
    """

    if len(counts) != len(random_basis):
        raise ValueError("The length of counts and random_basis must be the same.")

    new_random_basis = spreadout_random_basis(shots, random_basis)
    assert len(new_random_basis) == len(random_basis) * shots, (
        "The length of new_random_basis must be equal to len(random_basis) * shots."
        + f"len(new_random_basis): {len(new_random_basis)}, "
        + f"len(random_basis) * shots: {len(random_basis) * shots}, "
        + f"len(random_basis): {len(random_basis)}, shots: {shots}."
    )
    new_counts = spreadout_counts(counts)
    assert len(new_counts) == len(random_basis) * shots, (
        "The length of new_counts must be equal to len(random_basis) * shots."
        + f"len(new_counts): {len(new_counts)}, "
        + f"len(random_basis) * shots: {len(random_basis) * shots}, "
        + f"len(random_basis): {len(random_basis)}, shots: {shots}."
    )

    return new_counts, new_random_basis
