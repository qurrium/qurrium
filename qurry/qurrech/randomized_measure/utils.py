"""EchoListenRandomized - Utility
(:mod:`qurry.qurrech.randomized_measure.utils`)

"""

from typing import Union, Optional, Literal

from ...process.utils import qubit_mapper
from ...exceptions import (
    RandomizedMeasureUnitaryOperatorNotFullCovering,
    OverlapComparisonSizeDifferent,
)


def create_config(
    actual_qubits: int,
    measure: Optional[Union[list[int], tuple[int, int], int]],
    unitary_loc: Optional[Union[tuple[int, int], int]],
    which_circuit: Literal["1", "2"],
):
    """Create the configuration for the randomized measure.

    Args:
        actual_qubits (int): The number of qubits in the circuit.
        measure (Optional[Union[list[int], tuple[int, int], int]]):
            The selected qubits for the measurement.
            If it is None, then it will return the mapping of all qubits.
            If it is int, then it will return the mapping of the last n qubits.
            If it is tuple, then it will return the mapping of the qubits in the range.
            If it is list, then it will return the mapping of the selected qubits.
        unitary_loc (Optional[Union[tuple[int, int], int]]):
            The range of the unitary operator.
        which_circuit (Literal["1", "2"]): Which circuit this configuration belongs to.

    Returns:
        tuple[dict[int, int], list[int], dict[int, int], list[int]]:
            A tuple containing:
            - registers_mapping:
                The mapping of the index of selected qubits to the index of the classical register.
            - qubits_measured:
                The list of qubits that are measured.
            - unitary_located_mapping:
                The mapping of the index of unitary operator to the index of the classical register.
            - measured_but_not_unitary_located:
                The list of qubits that are measured but not located in the unitary operator.
    """

    registers_mapping = qubit_mapper(actual_qubits, measure)
    qubits_measured = list(registers_mapping)
    unitary_located_mapping = qubit_mapper(actual_qubits, unitary_loc)
    assert list(unitary_located_mapping.values()) == list(
        range(len(unitary_located_mapping))
    ), f"The unitary_located_mapping_{which_circuit} should be continuous."
    measured_but_not_unitary_located = [
        qi for qi in qubits_measured if qi not in unitary_located_mapping
    ]

    return (
        registers_mapping,
        qubits_measured,
        unitary_located_mapping,
        measured_but_not_unitary_located,
    )


MSG_OVERLAPPING_GIVEN = (
    "When the number of qubits in two circuits is not the same, "
    + "the {} of two circuits should be specified."
)


def overlapping_given_check(
    actual_qubits_1: int,
    actual_qubits_2: int,
    measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
    measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
    unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
    unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
):
    """Check whether the two circuits have overlapping qubits.

    Args:
        actual_qubits_1 (int): The number of qubits in the first circuit.
        actual_qubits_2 (int): The number of qubits in the second circuit.
            measure_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The selected qubits for the measurement for the first quantum circuit.
                If it is None, then it will return the mapping of all qubits.
                If it is int, then it will return the mapping of the last n qubits.
                If it is tuple, then it will return the mapping of the qubits in the range.
                If it is list, then it will return the mapping of the selected qubits.
                Defaults to None.
            measure_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The selected qubits for the measurement for the second quantum circuit.
                If it is None, then it will return the mapping of all qubits.
                If it is int, then it will return the mapping of the last n qubits.
                If it is tuple, then it will return the mapping of the qubits in the range.
                If it is list, then it will return the mapping of the selected qubits.
                Defaults to None.
            unitary_loc_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator for the first quantum circuit.
                Defaults to None.
            unitary_loc_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
                The range of the unitary operator for the second quantum circuit.
                Defaults to None.

    Raises:
        ValueError: If the number of qubits in the two circuits is not the same
            and the measure range or unitary location is not specified.
    """
    if actual_qubits_1 != actual_qubits_2:
        if any([measure_1 is None, measure_2 is None]):
            raise ValueError(MSG_OVERLAPPING_GIVEN.format("measure range"))
        if any([unitary_loc_1 is None, unitary_loc_2 is None]):
            raise ValueError(MSG_OVERLAPPING_GIVEN.format("unitary location"))


NSG_OVERLAPPING_SIZE = (
    "The qubits number of {} in two circuits should be the same, "
    + "but got different number of qubits measured."
    + "Got circuit 1: {} {} and circuit 2: {} {}."
)


def overlapping_size_check(
    qubits_measured_1: list[int],
    qubits_measured_2: list[int],
    unitary_located_mapping_1: dict[int, int],
    unitary_located_mapping_2: dict[int, int],
):
    """Check whether the size of the qubits measured and unitary located mapping are the same.

    Args:
        qubits_measured_1 (list[int]): The qubits measured in the first circuit.
        qubits_measured_2 (list[int]): The qubits measured in the second circuit.
        unitary_located_mapping_1 (dict[int, int]):
            The unitary located mapping in the first circuit.
        unitary_located_mapping_2 (dict[int, int]):
            The unitary located mapping in the second circuit.

    Raises:
        OverlapComparisonSizeDifferent:
            If the size of the qubits measured or unitary located mapping
            in the two circuits are different.
    """

    if len(qubits_measured_1) != len(qubits_measured_2):
        raise OverlapComparisonSizeDifferent(
            NSG_OVERLAPPING_SIZE.format(
                "measuring range",
                len(qubits_measured_1),
                qubits_measured_1,
                len(qubits_measured_2),
                qubits_measured_2,
            )
        )
    if len(unitary_located_mapping_1) != len(unitary_located_mapping_2):
        raise OverlapComparisonSizeDifferent(
            NSG_OVERLAPPING_SIZE.format(
                "unitary location",
                len(unitary_located_mapping_1),
                unitary_located_mapping_1,
                len(unitary_located_mapping_2),
                unitary_located_mapping_2,
            )
        )


MSG_FULL_COVER = (
    "Some qubits {} are measured "
    + "but not random unitary located in {} circuit. {}: {}, {}: {} "
    + "If you are sure about this, "
    + "you can set `unitary_loc_not_cover_measure=True` "
    + "to close this warning."
)


def unitary_full_cover_check(
    unitary_loc_not_cover_measure: bool,
    measured_but_not_unitary_located_1: list[int],
    measured_but_not_unitary_located_2: list[int],
    measure_1: Optional[Union[list[int], tuple[int, int], int]] = None,
    measure_2: Optional[Union[list[int], tuple[int, int], int]] = None,
    unitary_loc_1: Optional[Union[tuple[int, int], int]] = None,
    unitary_loc_2: Optional[Union[tuple[int, int], int]] = None,
):
    """Check whether the unitary operator covers the measurement.

    Args:
        unitary_loc_not_cover_measure (bool):
            If True, the unitary operator does not cover the measurement.
        measured_but_not_unitary_located_1 (list[int]):
            The qubits that are measured but not located in the first circuit.
        measured_but_not_unitary_located_2 (list[int]):
            The qubits that are measured but not located in the second circuit.
        measure_1 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the first quantum circuit.
            Defaults to None.
        measure_2 (Optional[Union[list[int], tuple[int, int], int]], optional):
            The selected qubits for the measurement for the second quantum circuit.
            Defaults to None.
        unitary_loc_1 (Optional[Union[tuple[int, int], int]], optional):
            The range of the unitary operator for the first quantum circuit.
            Defaults to None.
        unitary_loc_2 (Optional[Union[tuple[int, int], int]], optional):
            The range of the unitary operator for the second quantum circuit.
            Defaults to None.

    Raises:
        RandomizedMeasureUnitaryOperatorNotFullCovering:
            If the unitary operator does not cover the measurement and
            `unitary_loc_not_cover_measure` is False.
    """

    if not unitary_loc_not_cover_measure:
        if measured_but_not_unitary_located_1:
            raise RandomizedMeasureUnitaryOperatorNotFullCovering(
                MSG_FULL_COVER.format(
                    measured_but_not_unitary_located_1,
                    "first",
                    "unitary_loc_1",
                    unitary_loc_1,
                    "measure_1",
                    measure_1,
                ),
            )
        if measured_but_not_unitary_located_2:
            raise RandomizedMeasureUnitaryOperatorNotFullCovering(
                MSG_FULL_COVER.format(
                    measured_but_not_unitary_located_2,
                    "second",
                    "unitary_loc_2",
                    unitary_loc_2,
                    "measure_2",
                    measure_2,
                ),
            )
