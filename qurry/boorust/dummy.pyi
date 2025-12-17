"""Boorust - Toolkits for Dummy Case (:mod:`qurry.boorust.dummy`)"""

from typing import Optional

# pylint:disable=unused-argument
def make_two_bit_str_32(bitlen: int, num: Optional[int] = None) -> list[str]:
    """Make a list of bit strings with length of `num`.

    Args:
        bitlen (int): bit string length.
        num (Optional[int]): The number of bit strings.

    Returns:
        list[str]: The list of bit strings.
    """

def make_two_bit_str_unlimit(bitlen: int) -> list[str]:
    """Make a list of bit strings with length of `num`.

    Args:
        bitlen (int): bit string length.
        num (Optional[int]): The number of bit strings.

    Returns:
        list[str]: The list of bit strings.
    """

def make_dummy_case_32(
    n_a: int,
    shot_per_case: int,
    bitstring_num: Optional[int] = None,
) -> dict[str, int]:
    """Make a dummy case for the experiment.

    Args:
        n_a (int): Number of qubits in subsystem A.
        shot_per_case (int): Number of shots per case.
        bitstring_num (Optional[int]): Maximum number of bits.

    Returns:
        dict[str, int]: The dummy case.
    """
