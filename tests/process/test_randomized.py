"""Test qurry.process.randomized_measure module."""

from typing import Union
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.process.utils import cycling_slice, randomized_availability, NUMERICAL_ERROR_TOLERANCE
from qurry.process.randomized_measure.entangled_entropy_v1.entangled_entropy import (
    entangled_entropy_core,
)
from qurry.process.randomized_measure.entangled_entropy.entangled_entropy_2 import (
    entangled_entropy_core_2,
)
from qurry.process.randomized_measure.wavefunction_overlap_v1.wavefunction_overlap import (
    overlap_echo_core,
)
from qurry.process.randomized_measure.wavefunction_overlap.wavefunction_overlap_2 import (
    overlap_echo_core_2,
)
from qurry.process.randomized_measure import (
    entangled_availability,
    purity_cell_availability,
    entangled_v1_availability,
    purity_cell_v1_availability,
    overlap_availability,
    echo_cell_availability,
    overlap_v1_availability,
    echo_cell_v1_availability,
)


FILE_LOCATION = os.path.join(os.path.dirname(__file__), "easy-dummy.json")


easy_dummy: dict[str, dict[str, int]] = quickRead(FILE_LOCATION)
large_dummy_list = [easy_dummy["0"] for _ in range(2)]
test_setup_core: list[
    tuple[int, list[dict[str, int]], Union[int, tuple[int, int], None], tuple[int, int]]
] = [
    (4096, large_dummy_list, 6, (0, 8)),
    (4096, large_dummy_list, (2, 8), (0, 8)),
    (4096, large_dummy_list, 7, (0, 8)),
    (4096, large_dummy_list, (0, 7), (0, 8)),
    (4096, large_dummy_list, (-2, 5), (0, 8)),
    (4096, large_dummy_list, (-5, -1), (0, 8)),
    (4096, large_dummy_list, (3, -2), (0, 8)),
    (4096, large_dummy_list, None, (0, 8)),
]


def test_availability():
    """Test the availability of the Rust backend for the entangled_entropy_core function."""

    for availability_item in [
        randomized_availability,
        entangled_availability,
        purity_cell_availability,
        entangled_v1_availability,
        purity_cell_v1_availability,
        overlap_availability,
        echo_cell_availability,
        overlap_v1_availability,
        echo_cell_v1_availability,
    ]:
        assert availability_item[1]["Rust"], (
            "Rust is not available." + f" Check the error: {availability_item[2]}"
        )


def selected_and_cycling_selected_making(
    absolute_range: tuple[int, int],
    selected_range: Union[int, tuple[int, int], None],
) -> tuple[list[int], list[int]]:
    """Make selected classical registers based on the range
    and selected classical registers by cycling.

    Args:
        absolute_range (tuple[int, int]):
            The absolute range of classical registers, where the first element is the start
            and the second element is the end (exclusive).
        selected_range (Union[int, tuple[int, int], None]):
            The selected classical registers range or a single integer.
            If None, all registers in the absolute range are selected.
            If an integer, it selects that many registers from the end of the absolute range.
            If a tuple, it selects registers in the specified range.

    Returns:
        tuple[list[int], list[int]]:
            A tuple containing two lists:
            - The first list contains the selected classical registers.
            - The second list contains the selected classical registers by cycling.
    """
    return sorted(
        list(range(*absolute_range))
        if selected_range is None
        else (
            [
                absolute_range[1] - i % absolute_range[1] - 1
                for i in range(
                    *(
                        selected_range
                        if selected_range[0] < selected_range[1]
                        else tuple(ci % absolute_range[1] for ci in selected_range)
                    )
                )
            ]
            if isinstance(selected_range, tuple)
            else list(range(selected_range))
        )
    ), sorted(
        list(range(absolute_range[1] - 1, absolute_range[0] - 1, -1))
        if selected_range is None
        else (
            cycling_slice(
                list(range(absolute_range[1] - 1, absolute_range[0] - 1, -1)),
                selected_range[0],
                selected_range[1],
            )
            if isinstance(selected_range, tuple)
            else list(range(selected_range))
        )
    )


@pytest.mark.parametrize(
    ["shots", "counts", "selected_range", "absolute_range"],
    test_setup_core,
)
def test_entangled_entropy_core(
    shots: int,
    counts: list[dict[str, int]],
    selected_range: Union[int, tuple[int, int], None],
    absolute_range: tuple[int, int],
):
    """Test the entangled_entropy_core function."""

    selected_classical_registers, cycling_selected = selected_and_cycling_selected_making(
        absolute_range, selected_range
    )

    core_returned: dict[
        str,
        Union[
            tuple[
                Union[dict[int, float], dict[int, np.float64]],
                tuple[int, int],
                tuple[int, int],
            ],
            tuple[dict[int, np.float64], list[int]],
        ],
    ] = {}
    core_returned["Python V1"] = entangled_entropy_core(
        shots, counts, selected_range, absolute_range, backend="Python"
    )[:-2]
    core_returned["Python"] = entangled_entropy_core_2(
        shots,
        counts,
        selected_classical_registers,
        backend="Python",
    )[:-2]
    core_returned["Rust V1"] = entangled_entropy_core(
        shots, counts, selected_range, absolute_range, backend="Rust"
    )[:-2]
    core_returned["Rust"] = entangled_entropy_core_2(
        shots,
        counts,
        selected_classical_registers,
        backend="Rust",
    )[:-2]

    comparison_target = [
        (k, np.average(np.array(list(v[0].values()))), ", ".join(str(vv) for vv in v[1:]))
        for k, v in core_returned.items()
    ]

    for (desc_01, result_01, info_01), (desc_02, result_02, info_02) in combinations(
        comparison_target, 2
    ):
        assert np.abs(result_01 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{desc_01} and {desc_02} results are not equal in entangled_entropy_core: "
            + f"{desc_01}: {result_01}, {desc_02}: {result_02} - "
            + f"{desc_01}: {info_01}, {desc_02}: {info_02}"
        )

    assert selected_classical_registers == cycling_selected, (
        f"selected_classical_registers: {selected_classical_registers} != "
        + f"selected_classical_registers_by_cycling: {cycling_selected}"
    )


@pytest.mark.parametrize(
    ["shots", "counts", "selected_range", "absolute_range"],
    test_setup_core,
)
def test_overlap_echo_core(
    shots: int,
    counts: list[dict[str, int]],
    selected_range: Union[int, tuple[int, int], None],
    absolute_range: tuple[int, int],
):
    """Test the overlap_echo_core function."""

    selected_classical_registers, cycling_selected = selected_and_cycling_selected_making(
        absolute_range, selected_range
    )

    core_returned: dict[
        str,
        Union[
            tuple[
                Union[dict[int, float], dict[int, np.float64]],
                tuple[int, int],
                tuple[int, int],
            ],
            tuple[dict[int, np.float64], list[int]],
        ],
    ] = {}
    core_returned["Python V1"] = overlap_echo_core(
        shots,
        counts,
        selected_range,
        absolute_range,
        backend="Python",
    )[:-2]
    core_returned["Python"] = overlap_echo_core_2(
        shots,
        counts,
        counts,
        selected_classical_registers,
        backend="Python",
    )[:-2]
    core_returned["Rust V1"] = overlap_echo_core(
        shots, counts, selected_range, absolute_range, backend="Rust"
    )[:-2]
    core_returned["Rust"] = overlap_echo_core_2(
        shots,
        counts,
        counts,
        selected_classical_registers,
        backend="Rust",
    )[:-2]

    comparison_target = [
        (k, np.average(np.array(list(v[0].values()))), ", ".join(str(vv) for vv in v[1:]))
        for k, v in core_returned.items()
    ]

    for (name_1, result_1, info_1), (name_2, result_2, info_2) in combinations(
        comparison_target, 2
    ):
        assert np.abs(result_1 - result_2) < NUMERICAL_ERROR_TOLERANCE, (
            f"{name_1} and {name_2} results are not equal in entangled_entropy_core: "
            + f"{name_1}: {result_1}, {info_1}. {name_2}: {result_2}, {info_2}"
        )

    assert selected_classical_registers == cycling_selected, (
        f"selected_classical_registers: {selected_classical_registers} != "
        + f"selected_classical_registers_by_cycling: {cycling_selected}"
    )
