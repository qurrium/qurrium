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


@pytest.mark.parametrize("test_items", test_setup_core)
def test_entangled_entropy_core(
    test_items: tuple[int, list[dict[str, int]], Union[int, tuple[int, int]], tuple[int, int]],
):
    """Test the entangled_entropy_core function."""

    selected_classical_registers = sorted(
        list(range(*test_items[3]))
        if test_items[2] is None
        else (
            [
                test_items[3][1] - i % test_items[3][1] - 1
                for i in range(
                    *(
                        test_items[2]
                        if test_items[2][0] < test_items[2][1]
                        else tuple(ci % test_items[3][1] for ci in test_items[2])
                    )
                )
            ]
            if isinstance(test_items[2], tuple)
            else list(range(test_items[2]))
        )
    )
    selected_classical_registers_by_cycling = sorted(
        list(range(test_items[3][1] - 1, test_items[3][0] - 1, -1))
        if test_items[2] is None
        else (
            cycling_slice(
                list(range(test_items[3][1] - 1, test_items[3][0] - 1, -1)),
                test_items[2][0],
                test_items[2][1],
            )
            if isinstance(test_items[2], tuple)
            else list(range(test_items[2]))
        )
    )
    py = entangled_entropy_core(*test_items, backend="Python")
    py_2 = entangled_entropy_core_2(
        test_items[0],
        test_items[1],
        selected_classical_registers,
        backend="Python",
    )
    rust = entangled_entropy_core(*test_items, backend="Rust")
    rust_2 = entangled_entropy_core_2(
        test_items[0],
        test_items[1],
        selected_classical_registers,
        backend="Rust",
    )

    py_result = np.average(np.array(list(py[0].values())))
    py_2_result = np.average(np.array(list(py_2[0].values())))
    rust_result = np.average(np.array(list(rust[0].values())))
    rust_2_result = np.average(np.array(list(rust_2[0].values())))

    comparison_target = [
        ("py_2", "Python", py_2_result, f"{py_2[1]}"),
        ("py", "Python V1", py_result, f"{py[1]}, {py[2]}"),
        ("rust_2", "Rust", rust_2_result, f"{rust_2[1]}"),
        ("rust", "Rust V1", rust_result, f"{rust[1]}, {rust[2]}"),
    ]
    for (
        name_01,
        desc_01,
        result_01,
        info_01,
    ), (
        name_02,
        desc_02,
        result_02,
        info_02,
    ) in combinations(comparison_target, 2):
        assert np.abs(result_01 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{desc_01} and {desc_02} results are not equal in entangled_entropy_core: "
            + f"{name_01}: {result_01}, {name_02}: {result_02} - "
            + f"{name_01}: {info_01}, {name_02}: {info_02}"
        )

    assert selected_classical_registers == selected_classical_registers_by_cycling, (
        f"selected_classical_registers: {selected_classical_registers} != "
        + f"selected_classical_registers_by_cycling: {selected_classical_registers_by_cycling}"
    )


@pytest.mark.parametrize("test_items", test_setup_core)
def test_overlap_echo_core(
    test_items: tuple[int, list[dict[str, int]], Union[int, tuple[int, int]], tuple[int, int]],
):
    """Test the overlap_echo_core function."""

    selected_classical_registers = sorted(
        list(range(*test_items[3]))
        if test_items[2] is None
        else (
            [
                test_items[3][1] - i % test_items[3][1] - 1
                for i in range(
                    *(
                        test_items[2]
                        if test_items[2][0] < test_items[2][1]
                        else tuple(ci % test_items[3][1] for ci in test_items[2])
                    )
                )
            ]
            if isinstance(test_items[2], tuple)
            else list(range(test_items[2]))
        )
    )
    selected_classical_registers_by_cycling = sorted(
        list(range(test_items[3][1] - 1, test_items[3][0] - 1, -1))
        if test_items[2] is None
        else (
            cycling_slice(
                list(range(test_items[3][1] - 1, test_items[3][0] - 1, -1)),
                test_items[2][0],
                test_items[2][1],
            )
            if isinstance(test_items[2], tuple)
            else list(range(test_items[2]))
        )
    )
    py = overlap_echo_core(*test_items, backend="Python")
    py_2 = overlap_echo_core_2(
        test_items[0],
        test_items[1],
        test_items[1],
        selected_classical_registers,
        backend="Python",
    )
    rust = overlap_echo_core(*test_items, backend="Rust")
    rust_2 = overlap_echo_core_2(
        test_items[0],
        test_items[1],
        test_items[1],
        selected_classical_registers,
        backend="Rust",
    )

    py_result = np.average(np.array(list(py[0].values())))
    rust_result = np.average(np.array(list(rust[0].values())))
    py_2_result = np.average(np.array(list(py_2[0].values())))
    rust_2_result = np.average(np.array(list(rust_2[0].values())))

    comparison_target = [
        ("py_2", "Python", py_2_result, f"{py_2[1]}"),
        ("py", "Python V1", py_result, f"{py[1]}, {py[2]}"),
        ("rust_2", "Rust", rust_2_result, f"{rust_2[1]}"),
        ("rust", "Rust V1", rust_result, f"{rust[1]}, {rust[2]}"),
    ]
    for (
        name_01,
        desc_01,
        result_01,
        info_01,
    ), (
        name_02,
        desc_02,
        result_02,
        info_02,
    ) in combinations(comparison_target, 2):
        assert np.abs(result_01 - result_02) < NUMERICAL_ERROR_TOLERANCE, (
            f"{desc_01} and {desc_02} results are not equal in entangled_entropy_core: "
            + f"{name_01}: {result_01}, {name_02}: {result_02} - "
            + f"{name_01}: {info_01}, {name_02}: {info_02}"
        )

    assert selected_classical_registers == selected_classical_registers_by_cycling, (
        f"selected_classical_registers: {selected_classical_registers} != "
        + f"selected_classical_registers_by_cycling: {selected_classical_registers_by_cycling}"
    )
