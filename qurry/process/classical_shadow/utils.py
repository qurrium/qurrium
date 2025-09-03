"""Post Processing - Classical Shadow - Utilities (:mod:`qurry.process.classical_shadow.utils`)"""

from typing import Optional
import numpy as np


def generate_random_basis(
    snapshots: int,
    unitary_located: list[int],
    random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None,
) -> dict[int, dict[int, int]]:
    """Generate the random basis for the classical shadow.

    Args:
        snapshots (int): The number of snapshots.
        unitary_located (list[int]): The list of selected qubits.
        random_unitary_seeds (Optional[dict[int, dict[int, int]]]):
            The random unitary seeds.
            This argument only takes input as type of `dict[int, dict[int, int]]`.
            The first key is the index for the random unitary operator.
            The second key is the index for the qubit.

            .. code-block:: python

                {
                    0: {0: 1234, 1: 5678},
                    1: {0: 2345, 1: 6789},
                    2: {0: 3456, 1: 7890},
                }

            If you want to generate the seeds for all random unitary operator,
            you can use the function :func:`generate_random_unitary_seeds`
            in :mod:`qurry.process.randomized_measure.utils`.

            .. code-block:: python

                from qurry import generate_random_unitary_seeds

                random_unitary_seeds = generate_random_unitary_seeds(100, 2)

    Returns:
        dict[int, dict[int, int]]: The random basis.
    """
    if any(not isinstance(qi, int) for qi in unitary_located):
        raise ValueError("All qubits in unitary_located should be integers.")

    random_basis_placeholder = np.random.randint(
        0, 3, size=(snapshots, len(unitary_located))
    ).tolist()
    random_basis = {
        n_u_i: {
            n_u_qi: (
                random_basis_placeholder[n_u_i][seed_i]
                if random_unitary_seeds is None
                else int(np.random.default_rng(random_unitary_seeds[n_u_i][seed_i]).integers(0, 3))
            )
            for seed_i, n_u_qi in enumerate(unitary_located)
        }
        for n_u_i in range(snapshots)
    }
    return random_basis


def validate_random_basis(
    index: int, basis: dict[int, int], unitary_located: list[int]
) -> Optional[str]:
    """Validate the iteration of the random basis.

    Args:
        index (int): The index of the random basis item.
        basis (dict[int, int]): The random basis item.
        unitary_located (list[int]): The list of selected qubits.

    Returns:
        Optional[str]: The validation result.
    """
    if not isinstance(index, int):
        return f"Index '{index}' is not an integer, but '{type(index)}'."
    if not isinstance(basis, dict):
        return f"'{basis}' is not a dictionary."
    if not set(unitary_located).issubset(basis.keys()):
        return f"'selected_qubits' {unitary_located} are not in the random basis."
    if not all((isinstance(qi, int) and (0 <= q_basis < 3)) for qi, q_basis in basis.items()):
        return "All values should be integers in the range [0, 3) in the dictionary."
    return None


def check_random_basis(random_basis: dict[int, dict[int, int]], unitary_located: list[int]) -> bool:
    """Check if the random basis is valid.

    Args:
        random_basis (dict[int, dict[int, int]]): The random basis.
        unitary_located (list[int]): The list of selected qubits.

    Returns:
        bool: True if the random basis is valid.

    Raise:
        ValueError: If the random basis is invalid.
    """
    if not isinstance(random_basis, dict):
        raise ValueError("random_basis should be a dictionary.")
    if any(not isinstance(qi, int) for qi in unitary_located):
        raise ValueError("All qubits in unitary_located should be integers.")

    invalid_found = [
        (k, validate_random_basis(k, v, unitary_located)) for k, v in random_basis.items()
    ]
    invalid_dict = {k: v for k, v in invalid_found if v is not None}
    if invalid_dict:
        raise ValueError(f"Invalid random_basis: {invalid_dict}")

    return True
