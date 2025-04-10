"""EchoListenRandomized - Analysis
(:mod:`qurry.qurrech.randomized_measure.analysis`)

"""

from typing import Optional, NamedTuple, Iterable

from ...qurrium.analysis import AnalysisPrototype


class EchoListenRandomizedAnalysis(AnalysisPrototype):
    """The analysis of loschmidt echo."""

    __name__ = "EchoListenRandomizedAnalysis"

    class AnalysisInput(NamedTuple):
        """To set the analysis."""

        registers_mapping_1: dict[int, int]
        """The mapping of the classical registers with quantum registers.
        for the first quantum circuit.

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
        registers_mapping_2: dict[int, int]
        """The mapping of the classical registers with quantum registers.
        for the second quantum circuit.

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
        bitstring_mapping_1: Optional[dict[int, int]]
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
        bitstring_mapping_2: Optional[dict[int, int]]
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
        unitary_located_mapping_1: dict[int, int]
        """The range of the unitary operator for the first quantum circuit.

        .. code-block:: python
            {
                0: 0, # The quantum register 0 is used for the unitary operator 0.
                1: 1, # The quantum register 1 is used for the unitary operator 1.
                5: 2, # The quantum register 5 is used for the unitary operator 2.
                7: 3, # The quantum register 7 is used for the unitary operator 3.
            }

        The key is the index of the quantum register with the numerical order.
        The value is the index of the unitary operator with the numerical order.
        """
        unitary_located_mapping_2: dict[int, int]
        """The range of the unitary operator for the second quantum circuit.

        .. code-block:: python
            {
                0: 0, # The quantum register 0 is used for the unitary operator 0.
                1: 1, # The quantum register 1 is used for the unitary operator 1.
                5: 2, # The quantum register 5 is used for the unitary operator 2.
                7: 3, # The quantum register 7 is used for the unitary operator 3.
            }

        The key is the index of the quantum register with the numerical order.
        The value is the index of the unitary operator with the numerical order.
        """

    class AnalysisContent(NamedTuple):
        """The content of the analysis."""

        echo: float
        """The overlap value."""
        echoSD: float
        """The overlap standard deviation."""
        echoCells: dict[int, float]
        """The overlap of each single count."""
        num_classical_registers: int
        """The number of classical registers."""
        classical_registers: Optional[list[int]]
        """The list of the index of the selected classical registers."""
        classical_registers_actually: list[int]
        """The list of the index of the selected classical registers which is actually used."""
        # refactored
        counts_num: int
        """The number of first counts and second counts."""
        taking_time: float
        """The calculation time."""
        counts_used: Optional[Iterable[int]] = None
        """The index of the counts used.
        If not specified, then use all counts."""

        def __repr__(self):
            return f"AnalysisContent(echo={self.echo}, and others)"

    @property
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        return [
            "echoCells",
        ]
