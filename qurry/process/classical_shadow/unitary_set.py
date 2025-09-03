r"""Post Processing - Classical Shadow - Unitary Set
(:mod:`qurry.process.classical_shadow.unitary_set`)

The followings are unitary operators for our classical shadow implementation.
"""

from typing import Literal, Union
import functools as ft
import numpy as np
import numpy.typing as npt

from qiskit.circuit.gate import Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate

U_M_GATES: dict[Union[Literal[0, 1, 2], int], Gate] = {
    0: RXGate(np.pi / 2),
    1: RYGate(-np.pi / 2),
    2: RZGate(0),
}
r"""The :class:`~qiskit.circuit.library.Gate` objects
for the unitary operators :math:`U_m` in the classical shadow.

The set of unitary operators :math:`U_m` will represent by following dictionary:

- `0`: :math:`R_X(\frac{\pi}{2})`
- `1`: :math:`R_Y(-\frac{\pi}{2})`
- `2`: :math:`R_Z(0) = \mathbb{I}`

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    {
        0: Instruction(name='rx', num_qubits=1, num_clbits=0, params=[1.5707963267948966]),
        1: Instruction(name='ry', num_qubits=1, num_clbits=0, params=[-1.5707963267948966]),
        2: Instruction(name='rz', num_qubits=1, num_clbits=0, params=[0])
    }

"""

U_M_MATRIX: dict[Union[Literal[0, 1, 2], int], npt.NDArray[np.complex128]] = {
    0: np.array(
        [
            [np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)],
            [-1j * np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    ),
    1: np.array(
        [
            [np.cos(-np.pi / 4), -np.sin(-np.pi / 4)],
            [np.sin(-np.pi / 4), np.cos(-np.pi / 4)],
        ],
    ),
    2: np.array(
        [
            [np.exp(0), 0],
            [0, np.exp(0)],
        ]
    ),
}
r"""The :class:`~numpy.typing.NDArray[~numpy.complex128]` objects
for the unitary operators :math:`U_m` in the classical shadow.

The set of unitary operators :math:`U_m` will represent by following dictionary 
with the matrix representation:

- `0`: :math:`R_X(\frac{\pi}{2})`
- `1`: :math:`R_Y(-\frac{\pi}{2})`
- `2`: :math:`R_Z(0) = \mathbb{I}`

.. math::
    R_X(\frac{\pi}{2}) = \begin{pmatrix} \cos(\frac{\pi}{4}) & -i\sin(\frac{\pi}{4}) \\
    -i\sin(\frac{\pi}{4}) & \cos(\frac{\pi}{4}) \end{pmatrix} \\
    R_Y(-\frac{\pi}{2}) = \begin{pmatrix} \cos(-\frac{\pi}{4}) & -\sin(-\frac{\pi}{4}) \\
    \sin(-\frac{\pi}{4}) & \cos(-\frac{\pi}{4}) \end{pmatrix} \\
    R_Z(0) = \begin{pmatrix} e^{0} & 0 \\ 0 & e^{0} \end{pmatrix}

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    {
        0: array([
            [0.70710678 + 0.0j, 0.0 - 0.70710678j], 
            [0.0 - 0.70710678j, 0.70710678 + 0.0j]
        ]),
        1: array([
            [0.70710678, 0.70710678], 
            [-0.70710678, 0.70710678]
        ]),
        2: array([
            [1.0, 0.0], 
            [0.0, 1.0]
        ]),
    }

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


PRECOMPUTED_RHO_M_K_I = {
    (direction, b_k): (
        3 * U_M_MATRIX[direction].conj().T @ OUTER_PRODUCT[b_k] @ U_M_MATRIX[direction]
    )
    - IDENTITY
    for direction in [0, 1, 2]
    for b_k in ["0", "1"]
}
r"""Precomputed :math:`\rho_{mki}` matrix by

.. math::
    \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

where :math:`U_m` is the unitary operator, 
:math:`|b_k\rangle` is the k-th bitstring from counts on i-th qubit,
which is one of :math:`|0\rangle` and :math:`|1\rangle`.

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    {
        (0, "0"): array([
            [0.5 + 0.0j, 0.0 - 1.5j], 
            [0.0 + 1.5j, 0.5 + 0.0j]
        ]),
        (0, "1"): array([
            [0.5 + 0.0j, 0.0 + 1.5j], 
            [0.0 - 1.5j, 0.5 + 0.0j]
        ]),
        (1, "0"): array([
            [0.5, 1.5], 
            [1.5, 0.5]
        ]),
        (1, "1"): array([
            [0.5, -1.5], 
            [-1.5, 0.5]
        ]),
        (2, "0"): array([
            [2.0, 0.0], 
            [0.0, -1.0]
        ]),
        (2, "1"): array([
            [-1.0, 0.0], 
            [0.0, 2.0]
        ]),
    }

.. note::
    This is suggested by GitHub Copilot with Claude 3.7 Sonnet Thinking,
    which I never thought of.
"""


@ft.lru_cache(maxsize=1024)
def cached_rho_m_k_i_matrix(direction: int, bit: str) -> np.ndarray:
    r"""Cached :math:`\rho_{mki}` matrix from :const:`PRECOMPUTED_RHO_M_K_I`

    Args:
        direction (int): The direction of the shadow.
        bit (str): The bitstring.

    Returns:
        np.ndarray: The cached rho_m_k_i matrix.
    """
    return PRECOMPUTED_RHO_M_K_I[(direction, bit)]


PRECOMPUTED_RHO_M_K_I_2 = {
    direction * 10
    + int(b_k): (3 * U_M_MATRIX[direction].conj().T @ OUTER_PRODUCT[b_k] @ U_M_MATRIX[direction])
    - IDENTITY
    for direction in [0, 1, 2]
    for b_k in ["0", "1"]
}
r"""Precomputed :math:`\rho_{mki}` matrix, but use the integer as the key.

.. math::
    \rho_{mki} = 3 U_{mi}^{\dagger} |b_k \rangle\langle b_k| U_{mi} - \mathbb{I}

where :math:`U_m` is the unitary operator, 
:math:`|b_k\rangle` is the k-th bitstring from counts on i-th qubit,
which is one of :math:`|0\rangle` and :math:`|1\rangle`.

Here is the output of when you run the code in jupyter notebook:

.. code-block:: console

    {
        0: array([
            [0.5 + 0.0j, 0.0 - 1.5j], 
            [0.0 + 1.5j, 0.5 + 0.0j]
        ]),
        1: array([
            [0.5 + 0.0j, 0.0 + 1.5j], 
            [0.0 - 1.5j, 0.5 + 0.0j]
        ]),
        10: array([
            [0.5, 1.5], 
            [1.5, 0.5]
        ]),
        11: array([
            [0.5, -1.5], 
            [-1.5, 0.5]
        ]),
        20: array([
            [2.0, 0.0], 
            [0.0, -1.0]
        ]),
        21: array([
            [-1.0, 0.0], 
            [0.0, 2.0]
        ]),
    }

"""


@ft.lru_cache(maxsize=1024)
def cached_rho_m_k_i_matrix_2(index: int) -> np.ndarray:
    r"""Cached :math:`\rho_{mki}` matrix from :const:`PRECOMPUTED_RHO_M_K_I_2`

    Args:
        index (int): The index of the matrix.

    Returns:
        np.ndarray: The cached rho_m_k_i matrix.
    """
    return PRECOMPUTED_RHO_M_K_I_2[index]
