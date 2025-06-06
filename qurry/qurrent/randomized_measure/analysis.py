"""EntropyMeasureRandomized - Analysis
(:mod:`qurry.qurrent.randomized_measure.analysis`)

"""

from typing import Union, Optional, NamedTuple, Iterable, Literal

from ...qurrium.analysis import AnalysisPrototype


class EntropyMeasureRandomizedAnalysis(AnalysisPrototype):
    """The container for the analysis of :cls:`EntropyRandomizedExperiment`."""

    __name__ = "EntropyMeasureRandomizedAnalysis"

    class AnalysisInput(NamedTuple):
        """To set the analysis."""

        num_qubits: int
        """The number of qubits."""
        selected_qubits: list[int]
        """The selected qubits."""
        registers_mapping: dict[int, int]
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
        bitstring_mapping: Optional[dict[int, int]]
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
        shots: int
        """The number of shots."""
        unitary_located: Optional[list[int]] = None
        """The range of the unitary operator."""

    input: AnalysisInput

    class AnalysisContent(NamedTuple):
        """The content of the analysis."""

        purity: Optional[float] = None
        """The purity of the subsystem."""
        entropy: Optional[float] = None
        """The entanglement entropy of the subsystem."""
        puritySD: Optional[float] = None
        """The standard deviation of the purity of the subsystem."""
        entropySD: Optional[float] = None
        """The standard deviation of the entanglement entropy of the subsystem."""
        purityCells: Optional[dict[int, float]] = None
        """The purity of each cell of the subsystem."""
        # new added
        num_classical_registers: Optional[int] = None
        """The number of classical registers."""
        classical_registers: Optional[list[int]] = None
        """The list of the index of the selected classical registers."""
        classical_registers_actually: Optional[list[int]] = None
        """The list of the index of the selected classical registers which is actually used."""

        all_system_source: Optional[Union[str, Literal["independent", "null_counts"]]] = None
        """The name of source of all system.

        - independent: The all system is calculated independently.
        - null_counts: No counts exist.
        """

        purityAllSys: Optional[float] = None
        """The purity of the system."""
        entropyAllSys: Optional[float] = None
        """The entanglement entropy of the system."""
        puritySDAllSys: Optional[float] = None
        """The standard deviation of the purity of the system."""
        entropySDAllSys: Optional[float] = None
        """The standard deviation of the entanglement entropy of the system."""
        purityCellsAllSys: Optional[dict[int, float]] = None
        """The purity of each cell of the system."""
        # new added
        num_classical_registers_all_sys: Optional[int] = None
        """The number of classical registers of all system."""
        classical_registers_all_sys: Optional[list[int]] = None
        """The list of the index of the selected classical registers."""
        classical_registers_actually_all_sys: Optional[list[int]] = None
        """The list of the index of the selected classical registers which is actually used."""

        errorRate: Optional[float] = None
        """The error rate of the measurement from depolarizing error migigation calculated."""
        mitigatedPurity: Optional[float] = None
        """The mitigated purity of the subsystem."""
        mitigatedEntropy: Optional[float] = None
        """The mitigated entanglement entropy of the subsystem."""

        # refactored
        counts_num: Optional[int] = None
        """The number of counts."""
        taking_time: Optional[float] = None
        """The calculation time."""
        taking_time_all_sys: Optional[float] = None
        """The calculation time of the all system."""

        counts_used: Optional[Iterable[int]] = None
        """The index of the counts used.
        If not specified, then use all counts."""

        def __repr__(self):
            return f"AnalysisContent(purity={self.purity}, entropy={self.entropy}, and others)"

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return [
            "purityCells",
            "purityCellsAllSys",
        ]

    content: AnalysisContent
