from qiskit import QuantumCircuit

import warnings
from typing import Callable, Literal, Union, Hashable

from .qurryV4 import QurryV4
from .qurryV3 import QurryV3


def qubitSelector(
    num_qubits: int,
    degree: Union[int, tuple[int, int], None] = None,
    as_what: Literal['degree', 'unitary_set', 'measure range'] = 'degree',
) -> tuple[int]:
    """_summary_

    Args:
        num_qubits (int): _description_
        degree (Union[int, tuple[int, int], None], optional): _description_. Defaults to None.
        as_what (Literal[&#39;degree&#39;, &#39;unitary_set&#39;, &#39;measure range&#39;], optional): _description_. Defaults to 'degree'.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        tuple[int]: _description_
    """
    subsystem = [i for i in range(num_qubits)]
    item_range = ()

    if isinstance(degree, int):
        if degree > num_qubits:
            raise ValueError(
                f"The subsystem A includes {degree} qubits beyond {num_qubits} which the wave function has.")
        elif degree < 0:
            raise ValueError(
                f"The number of qubits of subsystem A has to be natural number.")

        item_range = (num_qubits-degree, num_qubits)
        subsystem = subsystem[num_qubits-degree:num_qubits]
    elif isinstance(degree, (tuple, list)):
        if len(degree) == 2:
            degParsed = [(d % num_qubits if d !=
                         num_qubits else num_qubits) for d in degree]
            item_range = (min(degParsed), max(degParsed))
            subsystem = subsystem[min(degParsed):max(degParsed)]
            print(
                f"| - Qubits: '{subsystem}' will be selected as {as_what}.")

        else:
            raise ValueError(
                f"Subsystem range is defined by only two integers, but there is {len(degree)} integers in '{degree}'.")

    else:
        raise ValueError("Degree of freedom is not given.")

    return item_range


def waveSelecter(
    qurry: Union[QurryV4, QurryV3],
    wave: Union[QuantumCircuit, any, None] = None,
) -> Hashable:
    """Select wave.

    Args:
        qurry (Union[QurryV4, QurryV3]): 
            The target qurry object.
        wave (Union[QuantumCircuit, int, None], optional): 
            The index of the wave function in `self.waves` or add new one to calaculation,
            then choose one of waves as the experiment material.
            If input is `QuantumCircuit`, then add and use it.
            If input is the key in `.waves`, then use it.
            If input is `None` or something illegal, then use `.lastWave'.
            Defaults to None.

    Returns:
        Hashable: wave
    """
    if isinstance(wave, QuantumCircuit):
        wave = qurry.addWave(wave)
        print(f"| Add new wave with key: {wave}")
    elif wave == None:
        wave = qurry.lastWave
        print(f"| Autofill will use '.lastWave' as key")
    else:
        try:
            qurry.waves[wave]
        except KeyError as e:
            warnings.warn(f"'{e}', use '.lastWave' as key")
            wave = qurry.lastWave

    return wave