"""Post Processing - Classical Shadow - Non Matrix Operation Process - Non MatMul Trace
(:mod:`qurry.process.classical_shadow.nomatop_process.nomatmul_trace`)

"""

from typing import Sequence, Iterable, Literal, Union
from functools import reduce
from itertools import combinations, batched
import multiprocessing as mp

# pylint:disable=no-name-in-module,import-error
from qurry.boorust.shadow import nomatmul_trace_sum_rust  # type: ignore


def rho_elt_compare(
    rho_ai_pauli: int, rho_bi_pauli: int, rho_ai_spin: int, rho_bi_spin: int
) -> float:
    """Calculate the trace of two classical shadows at a specific index.

    1. For random basis (Pauli basis) is not the same,
    the trace value will be `0.5`
    2. For random basis (Pauli basis) is the same,
    and the bitstring (spin) too, the trace value will be `5`
    3. For random basis (Pauli basis) is the same,
    but the bitstring (spin) is different, the trace value will be `-4`

    Args:
        rho_ai_pauli (int): Pauli basis from the first classical shadow. (X: 0, Y: 1, Z: 2)
        rho_bi_pauli (int): Pauli basis from the second classical shadow. (X: 0, Y: 1, Z: 2)
        rho_ai_spin (int): Spin value from the first classical shadow. (1, -1)
        rho_bi_spin (int): Spin value from the second classical shadow. (1, -1)

    Returns:
        float: The trace value calculated from the two shadows at the specified index.
    """
    return 0.5 if rho_ai_pauli != rho_bi_pauli else (5 if rho_ai_spin == rho_bi_spin else -4)


def get_trace(
    rho_a_pauli: Sequence[int],
    rho_a_spin: Sequence[int],
    rho_b_pauli: Sequence[int],
    rho_b_spin: Sequence[int],
    subsystem: Sequence[int],
) -> float:
    """Calculate the trace of two classical shadows.

    Args:
        rho_a_pauli (Sequence[int]): Pauli basis from the first shadow. (X: 0, Y: 1, Z: 2)
        rho_a_spin (Sequence[int]): Spin values from the first shadow. (1, -1)
        rho_b_pauli (Sequence[int]): Pauli basis from the second shadow. (X: 0, Y: 1, Z: 2)
        rho_b_spin (Sequence[int]): Spin values from the second shadow. (1, -1)
        subsystem (Sequence[int]): The subsystems.

    Returns:
        float: The trace value calculated from the two shadows.
    """
    if len(subsystem) == 0:
        return 1.0

    return reduce(
        lambda x, y: x * y,
        [
            rho_elt_compare(rho_a_pauli[i], rho_b_pauli[i], rho_a_spin[i], rho_b_spin[i])
            for i in subsystem
        ],
    )


def trace_calculation_unit(
    pauli_basis: Sequence[Sequence[int]],
    spin_outcome: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    list_of_pairs: Iterable[tuple[int, int]],
) -> float:
    """Calculate the trace for all pairs of classical shadows.

    Args:
        pauli_basis (Sequence[Sequence[int]]):
            The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (Sequence[Sequence[int]]):
            The list of spin outcomes. (1, -1)
        subsystem (Sequence[int]):
            The subsystems.
        list_of_pairs (Iterable[tuple[int, int]]):
            Pairs of indices for which to calculate the trace.

    Returns:
        float: The total trace value calculated from all pairs.
    """

    return sum(
        get_trace(pauli_basis[m1], spin_outcome[m1], pauli_basis[m2], spin_outcome[m2], subsystem)
        for m1, m2 in list_of_pairs
    )


def batch_make(num_of_samples: int):
    """Create a batched list of combinations for multiprocessing.

    Args:
        num_of_samples (int): The number of samples to create combinations from.

    Returns:
        A tuple containing:
            - A batched iterable of combinations.
            - The total number of batches.
    """
    return (
        batched(combinations(range(num_of_samples), 2), num_of_samples),
        num_of_samples // 2,
    )


def trace_calculation_unit_wrapper(
    args: tuple[
        Sequence[Sequence[int]], Sequence[Sequence[int]], Sequence[int], Iterable[tuple[int, int]]
    ],
) -> float:
    """Wrapper function for multiprocessing to calculate trace.

    Args:
        args (tuple[
            Sequence[Sequence[int]], Sequence[Sequence[int]],
            Sequence[int], Iterable[tuple[int, int]]
        ]):
            Tuple containing:
            - pauli_basis: The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
            - spin_outcome: The list of spin outcomes. (1, -1)
            - subsystem: The subsystems.
            - list_of_pairs: Pairs of indices for which to calculate the trace.
    Returns:
        float: The trace value calculated from the two shadows.
    """
    return trace_calculation_unit(*args)


def nomatmul_trace_sum_py(
    pauli_basis: Sequence[Sequence[int]],
    spin_outcome: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    multiprocessing: bool = True,
) -> float:
    """Perform the trace calculation for the given data and subsystems using Python.

    Args:
        pauli_basis (Sequence[Sequence[int]]):
            The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (Sequence[Sequence[int]]):
            The list of spin outcomes. (1, -1)
        subsystem (Sequence[int]):
            The subsystems.
        multiprocessing (bool):
            Whether to use multiprocessing for the calculation.

    Returns:
        float: The result of the trace calculation.
    """

    trace_m1_m2 = 0.0
    cpu_count = mp.cpu_count()

    all_combinations_split, all_combinations_split_num = batch_make(len(pauli_basis))
    chunksize = all_combinations_split_num // cpu_count // 4

    if multiprocessing or cpu_count > 1:
        with mp.Pool(cpu_count) as pool:
            # Using multiprocessing to parallelize the trace calculation
            results = pool.imap_unordered(
                trace_calculation_unit_wrapper,
                (
                    (pauli_basis, spin_outcome, subsystem, combination_item)
                    for combination_item in all_combinations_split
                ),
                chunksize=max(1, chunksize),
            )
            trace_m1_m2 += sum(results)
    else:
        # Without multiprocessing, calculate directly
        trace_m1_m2 += sum(
            trace_calculation_unit(pauli_basis, spin_outcome, subsystem, combination_item)
            for combination_item in all_combinations_split
        )

    return trace_m1_m2


NonMatMulTraceMethod = Union[
    Literal["nomatmul_trace_py", "nomatmul_trace_py_mp", "nomatmul_trace_rust"], str
]
"""The method to use for the trace calculation without matrix multiplication.

- "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
- "nomatmul_trace_py_mp": Use pure Python implementation with multiprocessing.
- "nomatmul_trace_rust": Use Rust implementation via PyO3.

The default method is "nomatmul_trace_rust", which is the fastest option.
"""

DEFAULT_NONMATMUL_TRACE_METHOD: NonMatMulTraceMethod = "nomatmul_trace_rust"
"""The default method to use for the trace calculation without matrix multiplication."""


def nomatmul_trace_sum(
    pauli_basis: Sequence[Sequence[int]],
    spin_outcome: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    trace_method: NonMatMulTraceMethod = DEFAULT_NONMATMUL_TRACE_METHOD,
) -> float:
    """Perform the trace calculation for the given data and subsystems.

    Args:
        pauli_basis (Sequence[Sequence[int]]):
            The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (Sequence[Sequence[int]]):
            The list of spin outcomes. (1, -1)
        subsystem (Sequence[int]):
            The subsystems.
        trace_method (NonMatMulTraceMethod):
            The method to use for the trace calculation.
            - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
            - "nomatmul_trace_py_mp": Use pure Python implementation with multiprocessing.
            - "nomatmul_trace_rust": Use Rust implementation via PyO3.
            Default is DEFAULT_NONMATMUL_TRACE_METHOD.

    Returns:
        float: The result of the trace calculation.
    """
    if trace_method == "nomatmul_trace_py":
        return nomatmul_trace_sum_py(pauli_basis, spin_outcome, subsystem, multiprocessing=False)
    if trace_method == "nomatmul_trace_py_mp":
        return nomatmul_trace_sum_py(pauli_basis, spin_outcome, subsystem, multiprocessing=True)
    if trace_method == "nomatmul_trace_rust":
        return nomatmul_trace_sum_rust(pauli_basis, spin_outcome, subsystem)
    raise ValueError(f"Unknown backend: {trace_method}")


def nomatmul_trace_core(
    pauli_basis: Sequence[Sequence[int]],
    spin_outcome: Sequence[Sequence[int]],
    subsystem: Sequence[int],
    trace_method: NonMatMulTraceMethod = DEFAULT_NONMATMUL_TRACE_METHOD,
) -> float:
    """Calculate the purity of the quantum state from the measurement data
    in string format without using matrix multiplication.

    Args:
        pauli_basis (Sequence[Sequence[int]]):
            The list of Pauli basis measurements. (X: 0, Y: 1, Z: 2)
        spin_outcome (Sequence[Sequence[int]]):
            The list of spin outcomes. (1, -1)
        subsystem (Sequence[int]):
            The subsystems.
        trace_method (NonMatMulTraceMethod):
            The method to use for the trace calculation.
            - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
            - "nomatmul_trace_py_mp": Use pure Python implementation with multiprocessing.
            - "nomatmul_trace_rust": Use Rust implementation via PyO3.
            Default is DEFAULT_NONMATMUL_TRACE_METHOD.

    Returns:
        float: The calculated purity of the quantum state.
    """
    num_of_samples = len(pauli_basis)
    if num_of_samples < 2:
        raise ValueError("At least two samples are required to calculate purity.")

    trace_m1_m2 = nomatmul_trace_sum(pauli_basis, spin_outcome, subsystem, trace_method)
    purity = 2 * trace_m1_m2 / (num_of_samples * (num_of_samples - 1))

    return purity
