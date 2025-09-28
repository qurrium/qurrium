"""Post Processing - Classical Shadow - Utilities - Snapshots/Shots Spreadout
(:mod:`qurry.process.classical_shadow.utils.spreadout`)

"""

from .random_basis import check_random_basis_array


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


def spreadout_random_basis(shots: int, random_basis_array: list[list[int]]) -> list[list[int]]:
    """Spreadout the random basis from multiple shots per snapshot to single shots counts.

    **Warning: We didn't use :func:`list.copy` in this function.**
    **For performance reasons, we avoid unnecessary copying of lists**
    **since they are totally the same all the time.**

    Args:
        random_basis_array (list[list[int]]):
            The random basis to be spreadout.


    Returns:
        list[list[int]]: The spreadout random basis.
    """

    return [single_random_basis for single_random_basis in random_basis_array for _ in range(shots)]


def spreadout(
    shots: int,
    counts: list[dict[str, int]],
    random_basis_array: list[list[int]],
) -> tuple[int, list[dict[str, int]], list[list[int]]]:
    """Spreadout the counts and random basis from multiple shots per snapshot
    to single shot per snapshot for classical shadow post-processing.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis (list[list[int]]):
            The random basis for classical shadow.

    Returns:
        tuple[list[dict[str, int]], list[list[int]]]:
            The spreadout shots, counts, and random basis.
    """
    check_random_basis_array(random_basis_array, len(counts), len(next(iter(counts[0].keys()))))

    new_random_basis = spreadout_random_basis(shots, random_basis_array)
    assert len(new_random_basis) == len(random_basis_array) * shots, (
        "The length of new_random_basis must be equal to len(random_basis) * shots."
        + f"len(new_random_basis): {len(new_random_basis)}, "
        + f"len(random_basis) * shots: {len(random_basis_array) * shots}, "
        + f"len(random_basis): {len(random_basis_array)}, shots: {shots}."
    )
    new_counts = spreadout_counts(counts)
    assert len(new_counts) == len(random_basis_array) * shots, (
        "The length of new_counts must be equal to len(random_basis) * shots."
        + f"len(new_counts): {len(new_counts)}, "
        + f"len(random_basis) * shots: {len(random_basis_array) * shots}, "
        + f"len(random_basis): {len(random_basis_array)}, shots: {shots}."
    )

    return 1, new_counts, new_random_basis
