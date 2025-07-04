"""Declaration - Types (:mod:`qurry.declare.type`)

Type declarations for Qurrium
"""

from typing import Union, Sequence

from qiskit import QuantumRegister
from qiskit.circuit.quantumcircuit import Qubit

QubitSpecifier = Union[
    Qubit,
    QuantumRegister,
    int,
    slice,
    Sequence[Union[Qubit, int]],
]
