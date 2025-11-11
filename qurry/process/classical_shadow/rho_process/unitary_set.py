"""Post Processing - Classical Shadow - Rho Process - Unitary Set
(:mod:`qurry.process.classical_shadow.rho_process.unitary_set`)

The followings are unitary operators for our classical shadow implementation.
"""

from typing import Optional, Sequence, Union, TypedDict
import functools as ft
import numpy as np
import numpy.typing as npt

from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.library import (
    HGate,
    IGate,
    RGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    UGate,
    XGate,
    YGate,
    ZGate,
)

from ...utils import BaseMethodEnum


def combine_gates(*gates: Gate) -> Gate:
    """Combine multiple single-qubit gates into a single gate.

    Args:
        *gates (Gate): Variable number of Gate objects to combine.

    Returns:
        Gate: The combined gate.
    """
    name = "-".join(gate.name for gate in gates)
    qc = QuantumCircuit(1, name=name)
    for gate in gates:
        qc.append(gate, [0])
    combined_gate = qc.to_gate(label=name)
    return combined_gate


def combine_gate_matrices(*gates: Gate) -> npt.NDArray[np.complex128]:
    """Combine the matrix representations of multiple gates.

    Args:
        *gates (Gate): Variable number of Gate objects to combine.

    Returns:
        npt.NDArray[np.complex_]: The combined matrix representation.
    """
    return ft.reduce(np.dot, [gate.to_matrix() for gate in reversed(gates)])


def get_basis_allow_gates() -> dict[str, Gate]:
    """Return a mapping of standard gate names to their corresponding gate objects."""

    lam = Parameter("λ")
    theta = Parameter("ϴ")
    phi = Parameter("φ")

    gates = [
        HGate(),
        IGate(),
        RGate(theta, phi),
        RXGate(theta),
        RYGate(theta),
        RZGate(phi),
        SGate(),
        SdgGate(),
        SXGate(),
        SXdgGate(),
        TGate(),
        TdgGate(),
        UGate(theta, phi, lam),
        XGate(),
        YGate(),
        ZGate(),
    ]
    return {gate.name: gate for gate in gates}


BASIS_ALLOW_GATES = get_basis_allow_gates()
r"""A dictionary mapping standard gate names to their corresponding Gate objects.
This dictionary includes commonly used single-qubit gates 
such as H, I, RX, RY, RZ, S, T, U, X, Y, and Z gates.
"""


OUTER_PRODUCT: dict[str, npt.NDArray[np.int32]] = {
    "0": np.array(
        [
            [1, 0],
            [0, 0],
        ]
    ),
    "1": np.array(
        [
            [0, 0],
            [0, 1],
        ]
    ),
}
r"""The :class:`~numpy.typing.NDArray[~numpy.int32]` objects 
for the outer product of :math:`|0\rangle` and :math:`|1\rangle`.

The set of outer product will represent by following dictionary
with the matrix representation:

- `0`: :math:`|0\rangle\langle0|`
- `1`: :math:`|1\rangle\langle1|`

.. math::
    |0\rangle\langle0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \\
    |1\rangle\langle1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    {
        0: array([
            [1, 0], 
            [0, 0]
        ]),
        1: array([
            [0, 0], 
            [0, 1]
        ]),
    }
"""

IDENTITY: npt.NDArray[np.int32] = np.array(
    [
        [1, 0],
        [0, 1],
    ],
)
r"""The :class:`~numpy.typing.NDArray[~numpy.int32]` objects for the identity matrix.

It's just :math:`\mathbb{I} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`.

What a simple matrix!

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    array([
        [1, 0],
        [0, 1]
    ])

"""


class ShadowRandomBasisData(TypedDict):
    """TypedDict for exporting and loading ShadowRandomBasis data."""

    name: str
    """The name of the ShadowRandomBasis."""
    gate_name_and_params: Sequence[list[tuple[str, list[float]]]]
    """The gate names and parameters for each basis."""


class ShadowRandomBasis:
    """Class for handling random basis selection for classical shadows."""

    @staticmethod
    def validate_basis_gates(basis_gates: tuple[Gate, ...]) -> None:
        """Validate that all gates in the basis_gates tuple are allowed.

        Args:
            basis_gates (tuple[Gate, ...]): Tuple of Gate objects to validate.

        Raises:
            ValueError: If any gate in basis_gates is not in BASIS_ALLOW_GATES.
        """
        if not isinstance(basis_gates, tuple):
            raise ValueError(
                "Each basis_gates must be a tuple of Gate objects to keep immutability."
            )
        if len(basis_gates) < 1:
            raise ValueError("Each basis_gates tuple must contain at least one Gate object.")

        if any(not isinstance(gate, Gate) for gate in basis_gates):
            raise ValueError("All elements in basis_gates tuples must be Gate objects.")
        if any(not hasattr(gate, "__array__") for gate in basis_gates):
            raise ValueError("All Gate objects must have a matrix representation.")
        if any(isinstance(param, Parameter) for gate in basis_gates for param in gate.params):
            raise ValueError(
                "All Gate objects must have concrete parameter values, not Parameters."
            )

        for gate in basis_gates:
            if gate.name not in BASIS_ALLOW_GATES:
                raise ValueError(
                    f"Gate '{gate.name}' is not allowed. "
                    f"Allowed gates are: {list(BASIS_ALLOW_GATES.keys())}."
                )
            if not isinstance(gate, BASIS_ALLOW_GATES[gate.name].__class__):
                raise ValueError("All elements in basis_gates must be Gate objects.")

    def __init__(
        self,
        *,
        basis_0_gates: tuple[Gate, ...],
        basis_1_gates: tuple[Gate, ...],
        basis_2_gates: tuple[Gate, ...],
        name: Optional[str] = None,
    ) -> None:
        """Initialize the ShadowRandomBasis with specified basis gates.

        Args:
            basis_0_gates (tuple[Gate, ...]): Tuple of Gate objects representing the measurement bases.
        """
        tmp_gates = (basis_0_gates, basis_1_gates, basis_2_gates)

        for basis_gates_tuple in tmp_gates:
            self.validate_basis_gates(basis_gates_tuple)

        self._gates_tuple = tmp_gates

        self._gates = (
            combine_gates(*self._gates_tuple[0]),
            combine_gates(*self._gates_tuple[1]),
            combine_gates(*self._gates_tuple[2]),
        )
        self._matrices = (
            combine_gate_matrices(*self._gates_tuple[0]),
            combine_gate_matrices(*self._gates_tuple[1]),
            combine_gate_matrices(*self._gates_tuple[2]),
        )
        self._gate_name_and_params: tuple[
            list[tuple[str, list[float]]],
            list[tuple[str, list[float]]],
            list[tuple[str, list[float]]],
        ] = (
            [(gate.name, gate.params) for gate in self._gates_tuple[0]],
            [(gate.name, gate.params) for gate in self._gates_tuple[1]],
            [(gate.name, gate.params) for gate in self._gates_tuple[2]],
        )
        self._name = "_".join(self._gates[i].name for i in range(3)) if name is None else name

        self._precomputed_rho_m_k_i = {
            (direction, b_k): (
                3
                * self._matrices[direction].conj().T
                @ OUTER_PRODUCT[b_k]
                @ self._matrices[direction]
            )
            - IDENTITY
            for direction in [0, 1, 2]
            for b_k in ["0", "1"]
        }
        self._precomputed_rho_m_k_i_2 = {
            (direction * 10 + ord(b_k) - 48): self._precomputed_rho_m_k_i[(direction, b_k)]
            for direction in [0, 1, 2]
            for b_k in ["0", "1"]
        }

    @property
    def gates_tuple(self) -> tuple[tuple[Gate, ...], tuple[Gate, ...], tuple[Gate, ...]]:
        """Get the original tuples of Gate objects for each basis.

        Returns:
            The original tuples of Gate objects for each basis.
        """
        return self._gates_tuple

    @property
    def gates(self) -> tuple[Gate, Gate, Gate]:
        """Get the combined Gate objects for each basis.

        Returns:
            The combined Gate objects for each basis.
        """
        return self._gates

    @property
    def matrices(
        self,
    ) -> tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
        """Get the combined matrix representations for each basis.

        Returns:
            The combined matrices for each basis.
        """
        return self._matrices

    @property
    def name(self) -> str:
        """Get the name of the ShadowRandomBasis.

        Returns:
            str: The name of the ShadowRandomBasis.
        """
        return self._name

    @property
    def precomputed_rho_m_k_i(self) -> dict[tuple[int, str], npt.NDArray[np.complex_]]:
        r"""Get the precomputed rho_m_k_i values.

        Precomputed :math:`\rho_{mki}` matrix by

        .. math::
            \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

        .. note::
            This is suggested by GitHub Copilot with Claude 3.7 Sonnet Thinking,
            which I never thought of :3.

        Returns:
            The precomputed rho_m_k_i values.
        """
        return self._precomputed_rho_m_k_i

    @ft.lru_cache
    def cached_precomputed_rho_m_k_i(self, direction: int, b_k: str) -> npt.NDArray[np.complex_]:
        r"""Get the cached precomputed rho_m_k_i value.

        .. math::
            \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

        where :math:`U_m` is the unitary operator,
        :math:`|b_k\rangle` is the k-th bitstring from counts on i-th qubit,
        which is one of :math:`|0\rangle` and :math:`|1\rangle`.

        Args:
            direction (int):
                The direction index (0, 1, or 2).
            b_k (str):
                The bitstring ('0' or '1').

        Returns:
            The cached precomputed rho_m_k_i value.
        """
        return self._precomputed_rho_m_k_i[(direction, b_k)]

    @property
    def precomputed_rho_m_k_i_2(self) -> dict[int, npt.NDArray[np.complex_]]:
        r"""Get the precomputed rho_m_k_i values.

        .. math::
            \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

        where :math:`U_m` is the unitary operator,
        :math:`|b_k\rangle` is the k-th bitstring from counts on i-th qubit,
        which is one of :math:`|0\rangle` and :math:`|1\rangle`.

        Returns:
            The precomputed rho_m_k_i values.
        """
        return self._precomputed_rho_m_k_i_2

    @ft.lru_cache
    def cached_precomputed_rho_m_k_i_2(self, key: int) -> npt.NDArray[np.complex_]:
        r"""Get the cached precomputed rho_m_k_i value.

        .. math::
            \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

        where :math:`U_m` is the unitary operator,
        :math:`|b_k\rangle` is the k-th bitstring from counts on i-th qubit,
        which is one of :math:`|0\rangle` and :math:`|1\rangle`.

        Args:
            key (int):
                The combined key (direction * 10 + b_k as int).

        Returns:
            The cached precomputed rho_m_k_i value.
        """
        return self._precomputed_rho_m_k_i_2[key]

    def export(self) -> ShadowRandomBasisData:
        """Export the ShadowRandomBasis data as a dictionary.

        Returns:
            ShadowRandomBasisData: A dictionary containing the ShadowRandomBasis data.
        """
        return {
            "name": self._name,
            "gate_name_and_params": self._gate_name_and_params,
        }

    @classmethod
    def load(cls, data: ShadowRandomBasisData) -> "ShadowRandomBasis":
        """Load the ShadowRandomBasis data from a dictionary.

        Args:
            data (ShadowRandomBasisData): A dictionary containing the ShadowRandomBasis data.

        Returns:
            ShadowRandomBasis: The loaded ShadowRandomBasis instance.
        """
        if "gate_name_and_params" not in data:
            raise ValueError("Data must contain 'gate_name_and_params' keys.")
        if len(data["gate_name_and_params"]) != 3:
            raise ValueError("gate_name_and_params must contain exactly three basis lists.")

        basis_gates = []
        for basis in data["gate_name_and_params"]:
            if not isinstance(basis, Sequence):
                raise ValueError("Each basis in gate_name_and_params must be a list.")
            if len(basis) < 1:
                raise ValueError("Each basis must contain at least one gate definition.")

            gates = []
            for gate_name, params in basis:
                gate_instance = BASIS_ALLOW_GATES.get(gate_name)
                if gate_instance is None:
                    raise ValueError(f"Gate '{gate_name}' is not recognized.")
                if len(params) != len(gate_instance.params):
                    raise ValueError(
                        f"Gate '{gate_name}' expects {len(gate_instance.params)} "
                        + f"parameters, but got {len(params)}."
                    )
                gates.append(gate_instance.__class__(*params, label=None))  # type: ignore[arg-type]
            basis_gates.append(tuple(gates))

        return cls(
            basis_0_gates=basis_gates[0],
            basis_1_gates=basis_gates[1],
            basis_2_gates=basis_gates[2],
            name=data.get("name", None),
        )

    def __repr__(self) -> str:
        return f"ShadowRandomBasis(name='{self._name}')"


BUILTIN_BASIS = {
    basis_obj.name: basis_obj
    for basis_obj in [
        ShadowRandomBasis(
            basis_0_gates=(RXGate(np.pi / 2),),
            basis_1_gates=(RYGate(-np.pi / 2),),
            basis_2_gates=(RZGate(0),),
            name="RX_RY_RZ",
        ),
        ShadowRandomBasis(
            basis_0_gates=(HGate(),),
            basis_1_gates=(SdgGate(), HGate()),
            basis_2_gates=(IGate(),),
            name="H_H-Sdg_I",
        ),
    ]
}
r"""Predefined :class:`ShadowRandomBasis` instances.

Here are the built-in basis sets:
- `RX_RY_RZ`: Uses :math:`R_X(\frac{\pi}{2})`, :math:`R_Y(-\frac{\pi}{2})`, and :math:`R_Z(0)` gates.
- `H_H-Sdg_I`: Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`, and Identity gates.
"""


class ShadowBasisMethod(BaseMethodEnum):
    """Enum for available unitary sets for classical shadow."""

    RX_RY_RZ = BUILTIN_BASIS["RX_RY_RZ"].name
    r"""Uses :math:`R_X(\frac{\pi}{2})`, :math:`R_Y(-\frac{\pi}{2})`, and :math:`R_Z(0)` gates.

    The basis of unitary operators is defined as:

    .. math::
        U = \{R_X(\frac{\pi}{2}), R_Y(-\frac{\pi}{4}), R_Z(0)\}

    The matrix representations are:

    .. math::
        R_X(\frac{\pi}{2}) = \begin{pmatrix} \cos(\frac{\pi}{4}) & -i\sin(\frac{\pi}{4}) \\
        -i\sin(\frac{\pi}{4}) & \cos(\frac{\pi}{4}) \end{pmatrix} \\
        R_Y(-\frac{\pi}{2}) = \begin{pmatrix} \cos(-\frac{\pi}{4}) & -\sin(-\frac{\pi}{4}) \\
        \sin(-\frac{\pi}{4}) & \cos(-\frac{\pi}{4}) \end{pmatrix} \\
        R_Z(0) = \begin{pmatrix} e^{0} & 0 \\ 0 & e^{0} \end{pmatrix}
        
    The console output of the matrices are:

    .. code-block:: console

        (array([[0.70710678+0.j        , 0.        -0.70710678j],
                [0.        -0.70710678j, 0.70710678+0.j        ]]),
         array([[ 0.70710678+0.j,  0.70710678+0.j],
                [-0.70710678+0.j,  0.70710678+0.j]]),
         array([[1.-0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]]))

    """

    H_H_SDG_I = BUILTIN_BASIS["H_H-Sdg_I"].name
    r"""Uses :math:`H`, :math:`H` followed by :math:`S^\dagger`, and Identity gates.
    
    The basis of unitary operators is defined as:

    .. math::
        U = \{H, HS^\dagger, I\}
    
    The matrix representations are:

    .. math::
        H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \\
        HS^\dagger = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -i \\ 1 & i \end{pmatrix} \\
        I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}

    The console output of the matrices are:

    .. code-block:: console

        (array([[ 0.70710678+0.j,  0.70710678+0.j],
                [ 0.70710678+0.j, -0.70710678+0.j]]),
         array([[0.70710678+0.j        , 0.        -0.70710678j],
                [0.70710678+0.j        , 0.        +0.70710678j]]),
         array([[1.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j]]))

    """

    @classmethod
    def get_default(cls) -> "ShadowBasisMethod":
        """Get the default basis method.

        Returns:
            ShadowBasisMethod: The default basis method.
        """
        return cls.RX_RY_RZ

    @classmethod
    def get_shadow_basis(
        cls, shadow_basis: Union["ShadowBasisMethod", str, ShadowRandomBasis]
    ) -> ShadowRandomBasis:
        """Get the ShadowRandomBasis instance for the given method.

        Args:
            shadow_basis (Union[ShadowBasisMethod, str, ShadowRandomBasis]):
                The shadow basis method or instance.

        Returns:
            ShadowRandomBasis: The corresponding ShadowRandomBasis instance.
        """

        if isinstance(shadow_basis, ShadowRandomBasis):
            return shadow_basis

        method = cls.from_string(shadow_basis) if isinstance(shadow_basis, str) else shadow_basis
        if method.value not in BUILTIN_BASIS:
            raise cls.value_error()
        return BUILTIN_BASIS[method.value]


ShadowBasisType = Union[ShadowBasisMethod, str, ShadowRandomBasis]
"""Type for shadow basis.
It can be either a :class:`ShadowBasisMethod` enum member, a string representing the enum member,
or a custom :class:`ShadowRandomBasis` instance.
"""


DEFAULT_SHADOW_BASIS: ShadowBasisType = ShadowBasisMethod.get_default()
"""The default shadow basis method."""
