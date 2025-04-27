"""Test qurry.process.classical_shadow module."""

from typing import TypedDict
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.qurrent.randomized_measure.utils import bitstring_mapping_getter
from qurry.process.classical_shadow import classical_shadow_complex, classical_shadow_availability

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "shadow-case.json")


class RawReadShadowCaseArguments(TypedDict):
    """TypedDict for shadow case arguments from JSON."""

    num_qubits: int
    selected_qubits: list[int]
    registers_mapping: dict[str, int]
    bitstring_mapping: dict[str, int]
    shots: int
    unitary_located: list[int]


class RawReadShadowCase(TypedDict):
    """TypedDict for shadow case data from JSON."""

    answer: dict[str, int]
    arguments: RawReadShadowCaseArguments
    random_unitary_ids: dict[str, dict[str, str]]
    counts: list[dict[str, int]]


class ShadowCaseArguments(TypedDict):
    """TypedDict for shadow case arguments."""

    num_qubits: int
    selected_qubits: list[int]
    registers_mapping: dict[int, int]
    bitstring_mapping: dict[int, int]
    shots: int
    unitary_located: list[int]


class ShadowCase(TypedDict):
    """TypedDict for shadow case."""

    answer: dict[str, int]
    arguments: ShadowCaseArguments
    random_unitary_ids: dict[int, dict[int, int]]
    counts: list[dict[str, int]]


raw_shadow_case_01: RawReadShadowCase = quickRead(FILE_LOCATION)
raw_shadow_cases: list[RawReadShadowCase] = [raw_shadow_case_01]
shadow_cases: list[ShadowCase] = [
    {
        "answer": shadow_case_tmp["answer"],
        "arguments": {
            "num_qubits": shadow_case_tmp["arguments"]["num_qubits"],
            "selected_qubits": shadow_case_tmp["arguments"]["selected_qubits"],
            "registers_mapping": {
                int(k): int(v) for k, v in shadow_case_tmp["arguments"]["registers_mapping"].items()
            },
            "bitstring_mapping": {
                int(k): int(v) for k, v in shadow_case_tmp["arguments"]["bitstring_mapping"].items()
            },
            "shots": shadow_case_tmp["arguments"]["shots"],
            "unitary_located": shadow_case_tmp["arguments"]["unitary_located"],
        },
        "random_unitary_ids": {
            int(k): {int(k2): int(v2) for k2, v2 in v.items()}
            for k, v in shadow_case_tmp["random_unitary_ids"].items()
        },
        "counts": shadow_case_tmp["counts"],
    }
    for shadow_case_tmp in raw_shadow_cases
]


@pytest.mark.parametrize("shadow_case", shadow_cases)
def test_shadow(shadow_case: ShadowCase):
    """Test the classical_shadow_complex function."""

    assert classical_shadow_availability[1]["Rust"] == "No", (
        "Rust is not available." + f" Check the error: {classical_shadow_availability[2]}"
    )

    _bitstring_mapping, final_mapping = bitstring_mapping_getter(
        shadow_case["counts"], shadow_case["arguments"]["registers_mapping"]
    )

    # Call the classical_shadow_complex function with the provided arguments
    result_py = classical_shadow_complex(
        shots=shadow_case["arguments"]["shots"],
        counts=shadow_case["counts"],
        random_unitary_um=shadow_case["random_unitary_ids"],
        selected_classical_registers=[
            final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
        ],
        backend="Python",
    )
    result_py_single_process = classical_shadow_complex(
        shots=shadow_case["arguments"]["shots"],
        counts=shadow_case["counts"],
        random_unitary_um=shadow_case["random_unitary_ids"],
        selected_classical_registers=[
            final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
        ],
        backend="Python",
        multiprocess=False,
    )
    result_py_einsum_ij_ij = classical_shadow_complex(
        shots=shadow_case["arguments"]["shots"],
        counts=shadow_case["counts"],
        random_unitary_um=shadow_case["random_unitary_ids"],
        selected_classical_registers=[
            final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
        ],
        backend="Python",
        method="hilbert_schmidt_inner_product",
    )
    result_py_single_process_einsum_ij_ij = classical_shadow_complex(
        shots=shadow_case["arguments"]["shots"],
        counts=shadow_case["counts"],
        random_unitary_um=shadow_case["random_unitary_ids"],
        selected_classical_registers=[
            final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
        ],
        backend="Python",
        multiprocess=False,
        method="hilbert_schmidt_inner_product",
    )
    # result_rust = classical_shadow_complex(
    #     shots=shadow_case["arguments"]["shots"],
    #     counts=shadow_case["counts"],
    #     random_unitary_um=shadow_case["random_unitary_ids"],
    #     selected_classical_registers=[
    #         final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
    #     ],
    #     backend="Rust",
    # )
    # result_rust_einsum_ij_ij = classical_shadow_complex(
    #     shots=shadow_case["arguments"]["shots"],
    #     counts=shadow_case["counts"],
    #     random_unitary_um=shadow_case["random_unitary_ids"],
    #     selected_classical_registers=[
    #         final_mapping[qi] for qi in shadow_case["arguments"]["selected_qubits"]
    #     ],
    #     backend="Rust",
    #     method="hilbert_schmidt_inner_product",
    # )

    compare_list = [
        (result_py, "result_py"),
        (result_py_single_process, "result_py_single_process"),
        (result_py_einsum_ij_ij, "result_py_einsum_ij_ij"),
        (result_py_single_process_einsum_ij_ij, "result_py_single_process_einsum_ij_ij"),
        # (result_rust, "result_rust"),
        # (result_rust_einsum_ij_ij, "result_rust_einsum_ij_ij"),
    ]

    # Compare the result with the expected answer
    for result_tmp, name in compare_list:
        assert np.abs(result_tmp["purity"] - shadow_case["answer"]["purity"]) < 1e-12, (
            "The result is not correct,"
            + f"{name}: {result_tmp["purity"]} != "
            + f"shadow_case['answer']: {shadow_case['answer']['purity']}"
        )
        assert np.abs(np.trace(result_tmp["expect_rho"]) - 1) < 1e-12, (
            "The trace of the expect_rho should be 1: " + f"{np.trace(result_tmp['expect_rho'])}."
        )

    for (result_tmp_1, name_1), (result_tmp_2, name_2) in combinations(compare_list, 2):
        assert np.abs(result_tmp_1["purity"] - result_tmp_2["purity"]) < 1e-12, (
            "The result is not correct,"
            + f"{name_1}: {result_tmp_1["purity"]} != {name_2}: {result_tmp_2["purity"]}"
        )
        assert (
            np.abs(np.trace(result_tmp_1["expect_rho"]) - np.trace(result_tmp_2["expect_rho"]))
            < 1e-12
        ), (
            "The trace of the expect_rho should be equal: "
            + f"{np.trace(result_tmp_1['expect_rho'])} != {np.trace(result_tmp_2['expect_rho'])}."
        )
