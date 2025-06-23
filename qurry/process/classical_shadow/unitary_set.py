r"""Post Processing - Classical Shadow - Unitary Set
(:mod:`qurry.process.classical_shadow.unitary_set`)

The followings are unitary operators for our classical shadow implementation.

.. math::
    U_M \in \{R_X(\frac{\pi}{2}), R_Y(-\frac{\pi}{2}), R_Z(0) = \mathbb{I} \}

And the set of unitary operators :math:`U_M` will represent by following dictionary.:

0. :math:`R_X(\frac{\pi}{2})`
1. :math:`R_Y(-\frac{\pi}{2})`
2. :math:`R_Z(0) = \mathbb{I}`

"""

from typing import Literal, Union
import numpy as np
import numpy.typing as npt

from qiskit.circuit.gate import Gate
from qiskit.circuit.library import RXGate, RYGate, RZGate

U_M_GATES: dict[Union[Literal[0, 1, 2], int], Gate] = {
    0: RXGate(np.pi / 2),
    1: RYGate(-np.pi / 2),
    2: RZGate(0),
}
r"""The :class:`qiskit.circuit.library.Gate` objects 
for the unitary operators :math:`U_M` in the classical shadow.

The set of unitary operators :math:`U_M` will represent by following dictionary.:

.. code-block:: text

    {
        0: :math:`R_X(\frac{\pi}{2})`,
        1: :math:`R_Y(-\frac{\pi}{2})`,
        2: :math:`R_Z(0) = \mathbb{I}`
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
r"""The :class:`NDArray[np.complex128]` objects 
for the unitary operators :math:`U_M` in the classical shadow.

The set of unitary operators :math:`U_M` will represent by following dictionary.:

.. code-block:: text

    {
        0: :math:`R_X(\frac{\pi}{2})`,
        1: :math:`R_Y(-\frac{\pi}{2})`,
        2: :math:`R_Z(0) = \mathbb{I}`
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
r"""The :class:`NDArray[np.int32]` objects 
for the outer product of :math:`|0\rangle` and :math:`|1\rangle`.

.. math::
    |0\rangle\langle0| = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
    |1\rangle\langle1| = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}

The set of outer product will represent by following dictionary.:

.. code-block:: text

    {
        "0": :math:`|0\rangle\langle0|`,
        "1": :math:`|1\rangle\langle1|`
    }
"""

IDENTITY: npt.NDArray[np.int32] = np.array(
    [
        [1, 0],
        [0, 1],
    ],
)
r"""The :class:`NDArray[np.int32]` objects for the identity matrix.

It's just :math:`\mathbb{I} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`.

What a simple matrix!
"""


PRECOMPUTED_RHO_M_K_I = {
    (direction, s_q): (
        3 * U_M_MATRIX[direction].conj().T @ OUTER_PRODUCT[s_q] @ U_M_MATRIX[direction]
    )
    - IDENTITY
    for direction in [0, 1, 2]
    for s_q in ["0", "1"]
}
r"""Precomputed :math:`\rho_{mki}` matrix.

.. note::
    This is suggested by GitHub Copilot with Claude 3.7 Sonnet Thinking,
    which I never thought of.
"""


PRECOMPUTED_RHO_M_K_I_2 = {
    direction * 10
    + int(s): (3 * U_M_MATRIX[direction].conj().T @ OUTER_PRODUCT[s] @ U_M_MATRIX[direction])
    - IDENTITY
    for direction in [0, 1, 2]
    for s in ["0", "1"]
}
r"""Precomputed :math:`\rho_{mki}` matrix.

But use the integer as the key.
"""
