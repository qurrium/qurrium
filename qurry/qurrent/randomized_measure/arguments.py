"""
===========================================================
EntropyMeasureRandomized - Arguments
(:mod:`qurry.qurrent.randomized_measure.arguments`)
===========================================================

"""

from typing import Optional, Union, Iterable
from collections.abc import Hashable
from dataclasses import dataclass

from qiskit import QuantumCircuit

from ...qurrium.experiment import ArgumentsPrototype
from ...process.randomized_measure.entangled_entropy import (
    PostProcessingBackendLabel,
)
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class EntropyMeasureRandomizedArguments(ArgumentsPrototype):
    """Arguments for the experiment."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    times: int = 100
    """The number of random unitary operator. 
    It will denote as `N_U` in the experiment name."""
    qubits_measured: Optional[list[int]] = None
    """The measure range."""
    registers_mapping: Optional[dict[int, int]] = None
    """The mapping of the classical registers of measurement with quantum registers.

    .. code-block:: python
        {
            0: 0, # The quantum register 0 is mapped to the classical register 0.
            1: 1, # The quantum register 1 is mapped to the classical register 1.
            5: 2, # The quantum register 5 is mapped to the classical register 2.
            7: 3, # The quantum register 7 is mapped to the classical register 3.
        }

    The key is the index of the quantum register with the numerical order.
    The value is the index of the classical register with the numerical order.
    """
    bitstring_mapping: Optional[dict[int, int]] = None
    """The mapping of the bitstring with the classical registers.
    When there are mulitple classical registers, 
    the bitstring is the concatenation of the classical registers with space on bitstring.
    For example, there are three registers with the size of 4, 4, and 6, 
    which the first six bits are for the randomized measurement.

    .. code-block:: python
        {'010000 0100 0001': 1024}
        # The bitstring is '010000 0100 0001'.
        # The last four bits are the first classical register.
        # The middle four bits are the second classical register.
        # The first six bits are the last classical register for the randomized measurement.

    So, the mapping will be like this.

    .. code-block:: python
        {
            0: 10, # The classical register 0 is mapped to the bitstring on the index 0.
            1: 11, # The classical register 0 is mapped to the bitstring on the index 1.
            2: 12, # The classical register 0 is mapped to the bitstring on the index 2.
            3: 13, # The classical register 0 is mapped to the bitstring on the index 3.
            4: 14, # The classical register 0 is mapped to the bitstring on the index 4.
            5: 15, # The classical register 0 is mapped to the bitstring on the index 5.
        }

    But, if there is only one classical register, 
    the bitstring will map to the classical register directly.

    .. code-block:: python
        {'010000': 1024}

    Will be like this.

    .. code-block:: python
        {
            0: 0, # The classical register 0 is mapped to the bitstring on the index 0.
            1: 1, # The classical register 0 is mapped to the bitstring on the index 1.
            2: 2, # The classical register 0 is mapped to the bitstring on the index 2.
            3: 3, # The classical register 0 is mapped to the bitstring on the index 3.
            4: 4, # The classical register 0 is mapped to the bitstring on the index 4.
            5: 5, # The classical register 0 is mapped to the bitstring on the index 5.
        }

    """
    actual_num_qubits: int = 0
    """The actual number of qubits."""
    unitary_located: Optional[list[int]] = None
    """The range of the unitary operator."""
    random_unitary_seeds: Optional[dict[int, dict[int, int]]] = None
    """The seeds for all random unitary operator.
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
    in :mod:`qurry.qurrium.utils.random_unitary`.

    .. code-block:: python
        from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds

        random_unitary_seeds = generate_random_unitary_seeds(100, 2)
    """


class EntropyMeasureRandomizedMeasureArgs(BasicArgs, total=False):
    """Output arguments for :meth:`output`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    times: int
    """The number of random unitary operator. 
    It will denote as `N_U` in the experiment name."""
    measure: Optional[Union[tuple[int, int], int, list[int]]]
    """The measure range."""
    unitary_loc: Optional[Union[tuple[int, int], int, list[int]]]
    """The range of the unitary operator."""
    unitary_loc_not_cover_measure: bool
    """Whether the range of the unitary operator is not cover the measure range."""
    random_unitary_seeds: Optional[dict[int, dict[int, int]]]
    """The seeds for all random unitary operator.
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
    in :mod:`qurry.qurrium.utils.random_unitary`.

    .. code-block:: python
        from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds

        random_unitary_seeds = generate_random_unitary_seeds(100, 2)
    """


class EntropyMeasureRandomizedOutputArgs(OutputArgs):
    """Output arguments for :meth:`output`."""

    times: int
    """The number of random unitary operator. 
    It will denote as `N_U` in the experiment name."""
    measure: Optional[Union[tuple[int, int], int, list[int]]]
    """The measure range."""
    unitary_loc: Optional[Union[tuple[int, int], int, list[int]]]
    """The range of the unitary operator."""
    unitary_loc_not_cover_measure: bool
    """Whether the range of the unitary operator is not cover the measure range."""
    random_unitary_seeds: Optional[dict[int, dict[int, int]]]
    """The seeds for all random unitary operator.
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
    in :mod:`qurry.qurrium.utils.random_unitary`.

    .. code-block:: python
        from qurry.qurrium.utils.random_unitary import generate_random_unitary_seeds

        random_unitary_seeds = generate_random_unitary_seeds(100, 2)
    """


class EntropyMeasureRandomizedAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of the analyze method."""

    selected_qubits: Optional[list[int]]
    """The selected qubits."""
    independent_all_system: bool
    """If True, then calculate the all system independently."""
    backend: PostProcessingBackendLabel
    """The backend for the process."""
    counts_used: Optional[Iterable[int]]
    """The index of the counts used."""


SHORT_NAME = "qurrent_randomized"
