from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.result import Result
from qiskit.providers.ibmq.managed import ManagedResults, IBMQManagedResultDataNotAvailable

import numpy as np
import warnings
from math import pi
from itertools import permutations
import time
from typing import Union, Optional, NamedTuple, Literal

from ..qurrium import Qurry
from ..tool import Configuration

# StringOperator V0.3.0 - Measuring Topological Phase - Qurstrop

stringOperatorLib = {
    'i': {
        'bound': {
            0: [''],
        },
        'filling': ['']
    },
    'zy': {
        'bound': {
            0: [],
            1: ['ry', -np.pi/2],
        },
        'filling': ['rx', np.pi/2]
    },
}

class StringOperator(Qurry):
    """StringOperator V0.3.0 of qurstrop

    - Reference:
        - Used in:
            Crossing a topological phase transition with a quantum computer - Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank, [PhysRevResearch.4.L022020](https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020)

        - `bibtex`:

```bibtex
@article{PhysRevResearch.4.L022020,
    title = {Crossing a topological phase transition with a quantum computer},
    author = {Smith, Adam and Jobst, Bernhard and Green, Andrew G. and Pollmann, Frank},
    journal = {Phys. Rev. Research},
    volume = {4},
    issue = {2},
    pages = {L022020},
    numpages = {8},
    year = {2022},
    month = {Apr},
    publisher = {American Physical Society},
    doi = {10.1103/PhysRevResearch.4.L022020},
    url = {https://link.aps.org/doi/10.1103/PhysRevResearch.4.L022020}
}
```
    """
    
    stringOperatorLib = {
        'i': {
            'bound': {
                0: [''],
            },
            'filling': ['']
        },
        'zy': {
            'bound': {
                0: [],
                1: ['ry', -np.pi/2],
            },
            'filling': ['rx', np.pi/2]
        },
    }

    class argdictCore(NamedTuple):
        expsName: str = 'exps'
        wave: Union[QuantumCircuit, any, None] = None,
        string: Literal['i', 'zy'] = 'i',
        # string: Literal[tuple(stringOperatorLib)] = 'i',
        i: Optional[int] = 1,
        k: Optional[int] = None,

    # Initialize
    def initialize(self) -> dict[str: any]:
        """Configuration to Initialize Qurrech.

        Returns:
            dict[str: any]: The basic configuration of `Qurrech`.
        """

        self._expsConfig = self.expsConfig(
            name="qurstropConfig",
        )
        self._expsBase = self.expsBase(
            name="qurstropBase",
            defaultArg={
                # Reault of experiment.
                'order': -100,
            },
        )
        self._expsHint = self.expsHint(
            name='qurstropBaseHint',
            hintContext={
                'order': 'The String Order Parameters.',
            },
        )
        self._expsMultiConfig = self.expsConfigMulti(
            name="qurstropConfigMulti",
        )
        self.shortName = 'qurstrop'
        self.__name__ = 'StringOperator'

        return self._expsConfig, self._expsBase

    """Arguments and Parameters control"""

    def paramsControlMain(
        self,
        expsName: str = 'exps',
        wave: Union[QuantumCircuit, any, None] = None,
        string: str = '1',
        i: Optional[int] = 0,
        k: Optional[int] = None,
        **otherArgs: any
    ) -> dict:
        """Handling all arguments and initializing a single experiment.

        Args:
            wave (Union[QuantumCircuit, int, None], optional): 
                The index of the wave function in `self.waves` or add new one to calaculation,
                then choose one of waves as the experiment material.
                If input is `QuantumCircuit`, then add and use it.
                If input is the key in `.waves`, then use it.
                If input is `None` or something illegal, then use `.lastWave'.
                Defaults to None.

            expsName (str, optional):
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.

            otherArgs (any):
                Other arguments.

        Raises:
            KeyError: Given `expID` does not exist.
            TypeError: When parameters are not all to be `int`.
            KeyError: The given parameters lost degree of freedom.".

        Returns:
            tuple[str, dict[str: any]]: Current `expID` and arguments.
        """

        # wave
        if isinstance(wave, QuantumCircuit):
            wave = self.addWave(wave)
            print(f"| Add new wave with key: {wave}")
        elif wave == None:
            wave = self.lastWave
            print(f"| Autofill will use '.lastWave' as key")
        else:
            try:
                self.waves[wave]
            except KeyError as e:
                warnings.warn(f"'{e}', use '.lastWave' as key")
                wave = self.lastWave

        numQubits = self.waves[wave].num_qubits
        # string order
        
        
        # i, k
        if i >= k:
            raise KeyError(f"'i ({i}) >= k ({k})' which is not allowed")
        # if 
        
        return {
            'wave': wave,
            'numQubit': numQubits,
            'string': string,
            'i': i,
            'k': k,
            'expsName': f"w={wave}-str={string}-i={i}-k={k}.{self.shortName}",
            **otherArgs,
        }

    """ Main Process: Circuit"""

    def circuitMethod(
        self,
    ) -> Union[QuantumCircuit, list[QuantumCircuit]]:
        """The method to construct circuit.
        Where should be overwritten by each construction of new measurement.

        Returns:
            Union[QuantumCircuit, list[QuantumCircuit]]: 
                The quantum circuit of experiment.
        """
        argsNow = self.now
        numQubits = self.waves[argsNow.wave].num_qubits

        qFunc = QuantumRegister(numQubits, 'q1')
        cMeas = ClassicalRegister(2, 'c1')
        qcExp = QuantumCircuit(qFunc, cMeas)

        qcExp.append(self.waveInstruction(
            wave=argsNow.wave,
            runBy=argsNow.runBy,
            backend=argsNow.backend,
        ), [qFunc[i] for i in range(numQubits)])

        qcExp.barrier()
        qcExp.measure(qFunc[i], cMeas[0])
        qcExp.measure(qFunc[j], cMeas[1])

        return [qcExp]

    """ Main Process: Data Import and Export"""

    """ Main Process: Job Create"""

    """ Main Process: Calculation and Result"""

    """ Main Process: Purity and Entropy"""

    @classmethod
    def quantity(
        cls,
        shots: int,
        result: Union[Result, ManagedResults],
        resultIdxList: Optional[list[int]] = None,
        numQubit: int = None,
        **otherArgs,
    ) -> tuple[dict, dict]:
        """Computing specific quantity.
        Where should be overwritten by each construction of new measurement.

        Returns:
            tuple[dict, dict]:
                Counts, purity, entropy of experiment.
        """

        if resultIdxList == None:
            resultIdxList = [i for i in range(numQubit*(numQubit-1))]
        elif isinstance(resultIdxList, list):
            if len(resultIdxList) > 1:
                ...
            elif len(resultIdxList) != numQubit*(numQubit-1):
                raise ValueError(
                    f"The element number of 'resultIdxList': {len(resultIdxList)} is different with 'N(N-1)': {times*2}.")
            else:
                raise ValueError(
                    f"The element number of 'resultIdxList': {len(resultIdxList)} needs to be more than 1 for 'StringOperator'.")
        else:
            raise ValueError("'resultIdxList' needs to be 'list'.")

        counts = []
        magnetsq = -100
        magnetsqCellList = []

        length = len(resultIdxList)
        idx = 0
        Begin = time.time()
        print(f"| Calculating magnetsq ...", end="\r")
        for i in resultIdxList:
            magnetsqCell = 0
            checkSum = 0
            print(
                f"| Calculating magnetsq on {i}" +
                f" - {idx}/{length} - {round(time.time() - Begin, 3)}s.", end="\r")

            try:
                allMeas = result.get_counts(i)
                counts.append(allMeas)
            except IBMQManagedResultDataNotAvailable as err:
                counts.append(None)
                print("| Failed Job result skip, index:", i, err)
                continue

            for bits in allMeas:
                checkSum += allMeas[bits]
                if (bits == '00') or (bits == '11'):
                    magnetsqCell += allMeas[bits]/shots
                else:
                    magnetsqCell -= allMeas[bits]/shots

            if checkSum != shots:
                raise ValueError(
                    f"'{allMeas}' may not be contained by '00', '11', '01', '10'.")

            magnetsqCellList.append(magnetsqCell)
            print(
                f"| Calculating magnetsq end - {idx}/{length}" +
                f" - {round(time.time() - Begin, 3)}s." +
                " "*30, end="\r")
            idx += 1
        print(
            f"| Calculating magnetsq end - {idx}/{length}" +
            f" - {round(time.time() - Begin, 3)}s.")

        magnetsq = (sum(magnetsqCellList) + numQubit)/(numQubit**2)

        quantity = {
            'magnetsq': magnetsq,
        }
        return counts, quantity

    """ Main Process: Main Control"""

    def measure(
        self,
        wave: Union[QuantumCircuit, any, None] = None,
        expsName: str = 'exps',
        **otherArgs: any
    ) -> dict:
        """

        Args:
            wave (Union[QuantumCircuit, int, None], optional):
                The index of the wave function in `self.waves` or add new one to calaculation,
                then choose one of waves as the experiment material.
                If input is `QuantumCircuit`, then add and use it.
                If input is the key in `.waves`, then use it.
                If input is `None` or something illegal, then use `.lastWave'.
                Defaults to None.

            expsName (str, optional):
                Naming this experiment to recognize it when the jobs are pending to IBMQ Service.
                This name is also used for creating a folder to store the exports.
                Defaults to `'exps'`.

            otherArgs (any):
                Other arguments.

        Returns:
            dict: The output.
        """
        return self.output(
            wave=wave,
            expsName=expsName,
            **otherArgs,
        )
