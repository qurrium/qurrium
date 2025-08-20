"""Test qurry.process.classical_shadow module."""

from typing import TypedDict
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.qurrent.randomized_measure.utils import bitstring_mapping_getter
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.classical_shadow import (
    classical_shadow_complex,
    ClassicalShadowComplex,
    classical_shadow_core_availability,
)
from qurry.process.classical_shadow.matrix_calcution import JAX_AVAILABLE

FILE_LOCATION = os.path.join(os.path.dirname(__file__), "shadow-case.json")

RHO_METHODS = ["numpy", "numpy_precomputed", "numpy_flatten"]
TRACE_METHODS = ["trace_of_matmul", "einsum_ij_ji", "einsum_aij_bji_to_ab_numpy"] + (
    ["einsum_aij_bji_to_ab_jax"] if JAX_AVAILABLE else []
)


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


class ClassicalShadowComplexExtended(ClassicalShadowComplex):
    """Extended ClassicalShadowComplex with expect_rho trace."""

    mean_of_rho_trace: np.complex128


raw_shadow_case_01: RawReadShadowCase = quickRead(FILE_LOCATION)
raw_shadow_cases: list[RawReadShadowCase] = [raw_shadow_case_01]
shadow_cases: list[
    tuple[dict[str, int], ShadowCaseArguments, dict[int, dict[int, int]], list[dict[str, int]]]
] = [
    (
        shadow_case_tmp["answer"],
        {
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
        {
            int(k): {int(k2): int(v2) for k2, v2 in v.items()}
            for k, v in shadow_case_tmp["random_unitary_ids"].items()
        },
        shadow_case_tmp["counts"],
    )
    for shadow_case_tmp in raw_shadow_cases
]


def test_availability():
    """Test the availability of the Rust backend for the entangled_entropy_core function."""

    for availability_item in [classical_shadow_core_availability]:
        assert availability_item[1]["Rust"], (
            "Rust is not available." + f" Check the error: {availability_item[2]}"
        )


@pytest.mark.parametrize(["answer", "arguments", "random_unitary_ids", "counts"], shadow_cases)
def test_shadow(
    answer: dict[str, int],
    arguments: ShadowCaseArguments,
    random_unitary_ids: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
):
    """Test the classical_shadow_complex function."""

    _bitstring_mapping, final_mapping = bitstring_mapping_getter(
        counts, arguments["registers_mapping"]
    )

    results: dict[str, ClassicalShadowComplexExtended] = {}

    # Call the classical_shadow_complex function with the provided arguments
    for rho_method in RHO_METHODS:
        for trace_method in TRACE_METHODS:
            tmp = classical_shadow_complex(
                shots=arguments["shots"],
                counts=counts,
                random_basis=random_unitary_ids,
                selected_classical_registers=[
                    final_mapping[qi] for qi in arguments["selected_qubits"]
                ],
                rho_method=rho_method,
                trace_method=trace_method,
            )
            results[rho_method + "." + trace_method] = {
                "mean_of_rho_trace": np.trace(tmp["mean_of_rho"]),
                **tmp,
            }

    # Compare the result with the expected answer
    for (name_1, result_1), (name_2, result_2) in combinations(results.items(), 2):
        assert np.abs(result_1["purity"] - result_2["purity"]) < NUMERICAL_ERROR_TOLERANCE, (
            "The result is not correct,"
            + f"{name_1}: {result_1['purity']} != {name_2}: {result_2['purity']}"
        )
        assert (
            np.abs(result_1["mean_of_rho_trace"] - result_2["mean_of_rho_trace"])
            < NUMERICAL_ERROR_TOLERANCE
        ), (
            "The trace of the expect_rho should be equal: "
            + f"{result_1['mean_of_rho_trace']} != {result_2['mean_of_rho_trace']}."
        )

    for name_1, result_1 in results.items():
        assert np.abs(result_1["purity"] - answer["purity"]) < NUMERICAL_ERROR_TOLERANCE, (
            "The result is not correct,"
            f"{name_1}: {result_1['purity']} != answer: {answer['purity']}"
        )
        assert np.abs(result_1["mean_of_rho_trace"] - 1) < NUMERICAL_ERROR_TOLERANCE, (
            "The trace of the expect_rho should be 1: " + f"{result_1['mean_of_rho_trace']}."
        )
