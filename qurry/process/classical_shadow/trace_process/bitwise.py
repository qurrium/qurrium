"""Post Processing - Classical Shadow - Trace Process - Bitwise Trace
(:mod:`qurry.process.classical_shadow.trace_process.bitwise`)

This derived from the implementation of Hsin-Yuan Huang
in `Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_.

The original implementation wrote in C++, so we have a Python conversion in
`harui2019/predicting-quantum-properties
<https://github.com/harui2019/predicting-quantum-properties>`_
by the author of Qurrium.

All following functions can be found in the Python conversion.

"""

from typing import Sequence, Union
from enum import Enum
import numpy as np

from ...utils import BaseMethodEnum


def count_trailing_zeros(n: int) -> int:
    """Calculate the number of trailing zeros in the binary representation of n.
    Equal to __builtin_ctzll(n) in C++.

    Args:
        n (int): The integer to count trailing zeros in.

    Returns:
        int: The number of trailing zeros in the binary representation of n.
    """
    if n == 0:
        return 64
    count = 0
    while (n & 1) == 0:
        n >>= 1
        count += 1
    return count


def c_non_id(c: int, subsystem_size: int) -> int:
    """Count the number of non-identity Pauli operators in the encoding.

    Description of the process:

    .. code-block:: text

        Gray Code: 0: 00, 1: 01, 2: 11, 3: 10
        Pauli: I: 0, X: 1, Y: 2, Z: 3
        => I: 00, X: 01, Y: 11, Z: 10

        Consider n = 8, max_encoding = 2 ** (2 * subsystem_size)
        For example, bitstring = 0b 00000000,
        and Pauli basis encoding c = 0b 0000 0001 1000 1100

            0000 0001 1000 1100 = II IX ZI YI
        &   1111 1111 1111 1111
        --------------------------------------
            0000 0001 1000 1100 = II IX ZI YI
        For only identity I: 00 keep 00 after AND with 11

        -> II IX ZI YI
        is not identity (I)
        -> 00 01 10 10
        sum(is not identity (I))
        -> non_id = 3

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        non_id = 0
        for i in range(subsystem_size):
            if ((c >> (2 * i)) & 3) != 0:  # 0b11 = 3
                non_id += 1

    Original implementaion in C++:

    .. code-block:: c++

        int nonId = 0;
        for(int i = 0; i < subsystem_size; i++)
            nonId += ((c >> (2 * i)) & 3) != 0;

    Args:
        c (int): The encoding to check.
        subsystem_size (int): The size of the subsystem.

    Returns:
        int: The number of non-identity Pauli operators.
    """

    return sum(((c >> (2 * i)) & 3) != 0 for i in range(subsystem_size))


def calculate_level_count(
    max_encoding: int, subsystem_size: int, all_number_of_outcomes: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Calculate the level count for a given encoding.

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        level_cnt = [0] * (subsystem_size + 1)
        level_ttl = [0] * (subsystem_size + 1)

        for c in range(max_encoding):
            non_id = 0
            for i in range(subsystem_size):
                if ((c >> (2 * i)) & 3) != 0:
                    non_id += 1

            if renyi_number_of_outcomes[c] >= 2:
                level_cnt[non_id] += 1
            level_ttl[non_id] += 1

    Original implementaion in C++:

    .. code-block:: c++

        int level_cnt[2 * subsystem_size], level_ttl[2 * subsystem_size];
        for(int i = 0; i < subsystem_size + 1; i++){
            level_cnt[i] = 0;
            level_ttl[i] = 0;
        }

        for(long long c = 0; c < (1 << (2 * subsystem_size)); c++){
            int nonId = 0;
            for(int i = 0; i < subsystem_size; i++){
                nonId += ((c >> (2 * i)) & 3) != 0;
            }
            if(renyi_number_of_outcomes[c] >= 2)
                level_cnt[nonId] ++;
            level_ttl[nonId] ++;
        }

    Args:
        max_encoding (int): The maximum encoding value.
        subsystem_size (int): The size of the subsystem.
        all_number_of_outcomes (Sequence[int]): The number of outcomes for each encoding.

    Returns:
        tuple[list[int], list[int]]: The level count and total count for each level.
    """
    non_id_values = np.array([c_non_id(c, subsystem_size) for c in range(max_encoding)])
    outcomes_array = np.array(all_number_of_outcomes)

    level_ttl = np.bincount(non_id_values, minlength=subsystem_size + 1).tolist()

    valid_mask = outcomes_array >= 2
    level_cnt = np.bincount(non_id_values[valid_mask], minlength=subsystem_size + 1).tolist()

    return level_cnt, level_ttl


def calculate_term_per_encoding(
    c: int,
    subsystem_size: int,
    sum_num_outcomes: int,
    binary_outcome: int,
    level_cnt: list[int],
    level_ttl: list[int],
) -> float:
    """Calculate the term for a given encoding in the entropy calculation.

    Original implementaion in Python by GitHub Copliot Claude Sonnet 4:

    .. code-block:: python

        if renyi_number_of_outcomes[c] <= 1:
            continue

        non_id = 0
        for i in range(subsystem_size):
            if ((c >> (2 * i)) & 3) != 0:
                non_id += 1

        if level_cnt[non_id] > 0:
            sum_squared = renyi_sum_of_binary_outcome[c] ** 2
            num_outcomes = renyi_number_of_outcomes[c]
            numerator = sum_squared - num_outcomes
            denominator = num_outcomes * (num_outcomes - 1)
            scale_factor = (
                level_ttl[non_id] / level_cnt[non_id] / (1 << subsystem_size)
            )
            term = numerator / denominator * scale_factor
            predicted_entropy += term

    Original implementaion in C++:

    .. code-block:: c++

        if(renyi_number_of_outcomes[c] <= 1) continue;

        int nonId = 0;
        for(int i = 0; i < subsystem_size; i++)
            nonId += ((c >> (2 * i)) & 3) != 0;

        predicted_entropy += ((double)1.0)
            / (renyi_number_of_outcomes[c] * (renyi_number_of_outcomes[c] - 1))
            * (
                renyi_sum_of_binary_outcome[c]
                * renyi_sum_of_binary_outcome[c]
                - renyi_number_of_outcomes[c]
            )
            / (1LL << subsystem_size)
            * level_ttl[nonId]
            / level_cnt[nonId];

    Args:
        c (int): The encoding to check.
        subsystem_size (int): The size of the subsystem.
        num_outcomes (int): The number of outcomes.
        sum_binary_outcome (int): The sum of binary outcome.
        level_cnt (list[int]): The count of levels.
        level_ttl (list[int]): The total time of levels.

    Returns:
        float: The calculated term for the given encoding.
    """
    if sum_num_outcomes <= 1:
        return 0.0

    non_id = c_non_id(c, subsystem_size)

    if level_cnt[non_id] <= 0:
        return 0.0

    return (
        (binary_outcome**2 - sum_num_outcomes)
        / (sum_num_outcomes * (sum_num_outcomes - 1))
        * (level_ttl[non_id] / level_cnt[non_id] / (1 << subsystem_size))
    )


def bitwise_core(
    pauli_basis: list[list[int]], spin_outcome: list[list[int]], subsystem: list[int]
) -> float:
    """Calculate the purity of a quantum system using bitwise operations.

    Args:
        pauli_basis (list[list[int]]): The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (list[list[int]]): The list of spin outcomes. (1, -1)
        subsystem (list[int]): The subsystems.

    Returns:
        float: The purity by bitwise.
    """
    if len(pauli_basis) != len(spin_outcome):
        raise ValueError(
            "Length mismatch: pauli_basis: "
            + f"{len(pauli_basis)} != spin_outcome: {len(spin_outcome)}"
        )

    subsystem_size = len(subsystem)
    max_encoding = 1 << (2 * subsystem_size)
    # Represent all possible comibinations of Pauli basis
    # Pauli: X: 1, Y: 2, Z: 3, I: 0 <-> X: 01, Y: 10, Z: 11, I: 00
    #
    # For example n = 8, 76543210, max_encoding = 2 ** (2 * subsystem_size)
    # For example c = 0b 0000 0001 1000 1100

    renyi_sum_of_binary_outcome = np.zeros(max_encoding, dtype=np.int64)
    # Record appearance times of all combinations of Pauli basis
    renyi_number_of_outcomes = np.zeros(max_encoding, dtype=np.int64)

    for single_pauli_base, single_spin_outcome in zip(pauli_basis, spin_outcome):
        encoding = 0  # encoding = 0b 0000 0000 0000 0000
        cumulative_outcome = 1

        renyi_sum_of_binary_outcome[0] += 1
        renyi_number_of_outcomes[0] += 1

        # Using gray code iteration over all 2^n possible outcomes
        # Gray Code: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
        # b in [1, 2^n-1]
        for b in range(1, 1 << subsystem_size):
            # For example n = 8, from 0b 0000 0001 to 0b 1111 1111
            # This is the bitstring from all possible measurement outcomes
            # ctz(01010101) -> 0, ctz(10000000) -> 7
            change_i = count_trailing_zeros(b)
            index_in_original_system = subsystem[change_i]
            # The clregs index

            cumulative_outcome *= single_spin_outcome[index_in_original_system]

            pauli_value = single_pauli_base[index_in_original_system]
            encoding ^= (pauli_value + 1) << (2 * change_i)
            # Gray Code: 0 -> 00, 1 -> 01, 2 -> 11, 3 -> 10
            # Pauli: I: 0, X: 1, Y: 2, Z: 3 <-> I: 00, X: 01, Y: 11, Z: 10
            # ----
            # For example ctz(01010101) -> 0 with X, ctz(10000000) -> 7 with Z
            #
            # ctz(01010101) -> 0 -> move 0  -> 0000 0000 0000 0001
            # ctz(10000000) -> 7 -> move 14 -> 1000 0000 0000 0000
            # ---
            # bitwise XOR, for example (0101) ^ (0011) = (0110)
            #
            # new_encoding_01: int = encoding ^ 0b 0000 0000 0000 0001
            # new_encoding_02: int = encoding ^ 0b 0000 0000 0000 0010

            renyi_sum_of_binary_outcome[encoding] += cumulative_outcome
            renyi_number_of_outcomes[encoding] += 1
            # binary_outcome[new_encoding_01] += (+1 or -1)
            # number_of_outcomes[new_encoding_01] += 1
            # binary_outcome[new_encoding_02] += (+1 or -1)
            # number_of_outcomes[new_encoding_02] += 1

    # Calculate level counts with optimized version
    level_cnt, level_ttl = calculate_level_count(
        max_encoding,
        subsystem_size,
        renyi_number_of_outcomes,  # type: ignore
    )

    return sum(
        calculate_term_per_encoding(
            c,
            subsystem_size,
            renyi_number_of_outcomes[c],
            renyi_sum_of_binary_outcome[c],
            level_cnt,
            level_ttl,
        )
        for c in range(max_encoding)
    )


class BitWiseTraceMethod(BaseMethodEnum, Enum):
    """The method to use for the trace calculation without matrix multiplication.

    - "bitwise_py": Use pure Python bitwise implementation.

    The default method is "bitwise_py", which is the fastest option.
    """

    BITWISE_PY = "bitwise_py"
    """Use pure Python bitwise implementation."""
    # BITWISE_RUST = "bitwise_rust"
    # """Use Rust implementation via PyO3."""

    @classmethod
    def get_default(cls) -> "BitWiseTraceMethod":
        """Get the default method for the trace calculation without matrix multiplication.

        Returns:
            BitWiseTraceMethod: The default method.
        """
        return cls.BITWISE_PY


BitWiseTraceMethodType = Union[BitWiseTraceMethod, str]
"""The method to use for the trace calculation with bitwise operations."""

DEFAULT_BITWISE_TRACE_METHOD: BitWiseTraceMethod = BitWiseTraceMethod.get_default()
"""The default method for the trace calculation with bitwise operations."""
