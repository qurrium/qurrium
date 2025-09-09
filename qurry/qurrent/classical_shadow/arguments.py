"""ShadowUnveil - Arguments
(:mod:`qurry.qurrent.classical_shadow.arguments`)

"""

from typing import Optional, Union, Iterable
from collections.abc import Hashable
from dataclasses import dataclass
import numpy as np

from qiskit import QuantumCircuit

from ...qurrium.experiment import ArgumentsPrototype
from ...process.classical_shadow import RhoMethod, ListTraceMethod, TraceMethod
from ...declare import BasicArgs, OutputArgs, AnalyzeArgs


@dataclass(frozen=True)
class ShadowUnveilArguments(ArgumentsPrototype):
    """Arguments for
    :class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`."""

    exp_name: str = "exps"
    """The name of the experiment.
    Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
    This name is also used for creating a folder to store the exports.
    Defaults to `'experiment'`."""
    snapshots: int = 100
    """The number of random unitary operator, previously called `times`
    It will denote as :math:`N_U` in the experiment name."""
    qubits_measured: Optional[list[int]] = None
    """The measure range."""
    registers_mapping: Optional[dict[int, int]] = None
    """The mapping of the classical registers with quantum registers.

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
    actual_num_qubits: int = 0
    """The actual number of qubits."""
    unitary_located: Optional[list[int]] = None
    """The range of the unitary operator."""
    random_basis: Optional[dict[int, dict[int, int]]] = None
    """The random basis for classical shadow.

    This argument only takes input as type of `dict[int, dict[int, int]]`.
    The first key is the index if snapshots.
    The second key is the index for the qubit.

    .. code-block:: python

        {
            0: {0: 1, 1: 0},
            1: {0: 2, 1: 1},
            2: {0: 0, 1: 2},
        }

    If you want to generate the seeds for all random unitary operator,
    you can use the function :func:`generate_random_basis` 
    in :mod:`qurry.process.classical_shadow.utils`.

    .. code-block:: python

        from qurry import generate_random_basis

        random_basis = generate_random_basis(100, [0, 1])
    """

    def __post_init__(self):
        if self.registers_mapping is not None:
            super().__setattr__(
                "registers_mapping", {int(k): int(v) for k, v in self.registers_mapping.items()}
            )

        if self.random_basis is not None:
            super().__setattr__(
                "random_basis",
                {
                    int(k): {int(k2): int(v2) for k2, v2 in v.items()}
                    for k, v in self.random_basis.items()
                },
            )


class ShadowUnveilMeasureArgs(BasicArgs, total=False):
    """Input fields for
    :meth:`~qurry.qurrent.classical_shadow.qurry.ShadowUnveil.measure`
    and :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiOutput`."""

    wave: Optional[Union[QuantumCircuit, Hashable]]
    """The key or the circuit to execute."""
    snapshots: int
    """The number of random unitary operator, previously called `times`
    It will denote as :math:`N_U` in the experiment name."""
    measure: Optional[Union[tuple[int, int], int, list[int]]]
    """The measure range."""
    unitary_loc: Optional[Union[tuple[int, int], int, list[int]]]
    """The range of the unitary operator."""
    unitary_loc_not_cover_measure: bool
    """Whether the range of the unitary operator is not cover the measure range."""
    random_basis: Optional[dict[int, dict[int, int]]]
    """The random basis for classical shadow.

    This argument only takes input as type of `dict[int, dict[int, int]]`.
    The first key is the index if snapshots.
    The second key is the index for the qubit.

    .. code-block:: python

        {
            0: {0: 1, 1: 0},
            1: {0: 2, 1: 1},
            2: {0: 0, 1: 2},
        }

    If you want to generate the seeds for all random unitary operator,
    you can use the function :func:`generate_random_basis` 
    in :mod:`qurry.process.classical_shadow.utils`.

    .. code-block:: python

        from qurry import generate_random_basis

        random_basis = generate_random_basis(100, [0, 1])
    """


class ShadowUnveilOutputArgs(OutputArgs):
    """Output arguments for
    :meth:`~qurry.qurrent.classical_shadow.qurry.ShadowUnveil.output`."""

    snapshots: int
    """The number of random unitary operator, previously called `times`
    It will denote as :math:`N_U` in the experiment name."""
    measure: Optional[Union[tuple[int, int], int, list[int]]]
    """The measure range."""
    unitary_loc: Optional[Union[tuple[int, int], int, list[int]]]
    """The range of the unitary operator."""
    unitary_loc_not_cover_measure: bool
    """Whether the range of the unitary operator is not cover the measure range."""
    random_basis: Optional[dict[int, dict[int, int]]]
    """The random basis for classical shadow.

    This argument only takes input as type of `dict[int, dict[int, int]]`.
    The first key is the index if snapshots.
    The second key is the index for the qubit.

    .. code-block:: python

        {
            0: {0: 1, 1: 0},
            1: {0: 2, 1: 1},
            2: {0: 0, 1: 2},
        }

    If you want to generate the seeds for all random unitary operator,
    you can use the function :func:`generate_random_basis` 
    in :mod:`qurry.process.classical_shadow.utils`.

    .. code-block:: python

        from qurry import generate_random_basis

        random_basis = generate_random_basis(100, [0, 1])
    """


class ShadowUnveilAnalyzeArgs(AnalyzeArgs, total=False):
    """The input of :meth:`~qurry.qurrium.qurrium.QurriumPrototype.multiAnalysis` and
    :meth:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment.analyze`.
    """

    selected_qubits: Optional[list[int]]
    """The selected qubits."""
    # estimation of given operators
    given_operators: Optional[list[np.ndarray[tuple[int, int], np.dtype[np.complex128]]]]
    """The list of the operators to estimate."""
    accuracy_prob_comp_delta: float
    """The accuracy probability for computing delta."""
    max_shadow_norm: Optional[float]
    """The maximum shadow norm of the given operators."""
    # other config
    rho_method: RhoMethod
    """The method to reconstruct the density matrix."""
    trace_method: TraceMethod
    """The method to compute the trace."""
    estimate_trace_method: ListTraceMethod
    """The method to estimate the trace."""
    counts_used: Optional[Iterable[int]]
    """The index of the counts used."""


SHORT_NAME = "qurshady_entropy"
"""The short name of
:class:`~qurry.qurrent.classical_shadow.experiment.ShadowUnveilExperiment`."""
