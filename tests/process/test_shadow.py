"""Test qurry.process.classical_shadow module."""

from typing import TypedDict
import os
from itertools import combinations
import pytest
import numpy as np

from qurry.capsule import quickRead
from qurry.qurrium.utils import bitstring_mapping_getter
from qurry.process.utils import NUMERICAL_ERROR_TOLERANCE
from qurry.process.classical_shadow import (
    classical_shadow_complex,
    ClassicalShadowComplex,
    classical_shadow_rho_process_availability,
    classical_shadow_matrix_availability,
    JAX_AVAILABLE,
    RhoMethod,
    RhoMethodType,
    TraceMethod,
    TraceMethodType,
    purity_value_kind,
    PurityValueKind,
)

FILE_LOCATIONS = [
    os.path.join(os.path.dirname(__file__), "shadow-case.json"),
    os.path.join(os.path.dirname(__file__), "shadow-case-hard.json"),
]

methods_by_kind: dict[PurityValueKind, list[tuple[RhoMethodType, TraceMethodType]]] = {}

for rho_method_tmp in RhoMethod.get_all_methods():
    for trace_method_tmp in TraceMethod.get_all_methods():
        if not JAX_AVAILABLE and trace_method_tmp == TraceMethod.EINSUM_AIJ_BJI_TO_AB_JAX.value:
            continue
        methods_by_kind.setdefault(purity_value_kind(rho_method_tmp, trace_method_tmp), []).append(
            (rho_method_tmp, trace_method_tmp)
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

    answer_multi_shots: dict[str, int]
    answer_single_shots: dict[str, int]
    answer_bitwise: dict[str, int]
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


raw_shadow_cases: list[RawReadShadowCase] = [quickRead(file_loc) for file_loc in FILE_LOCATIONS]


def unpacked_shadow_case(
    shadow_case: RawReadShadowCase,
) -> tuple[ShadowCaseArguments, dict[int, dict[int, int]], list[dict[str, int]]]:
    """Unpack the shadow case from RawReadShadowCase to ShadowCaseArguments."""
    return (
        {
            "num_qubits": shadow_case["arguments"]["num_qubits"],
            "selected_qubits": shadow_case["arguments"]["selected_qubits"],
            "registers_mapping": {
                int(k): int(v) for k, v in shadow_case["arguments"]["registers_mapping"].items()
            },
            "bitstring_mapping": {
                int(k): int(v) for k, v in shadow_case["arguments"]["bitstring_mapping"].items()
            },
            "shots": shadow_case["arguments"]["shots"],
            "unitary_located": shadow_case["arguments"]["unitary_located"],
        },
        {
            int(k): {int(k2): int(v2) for k2, v2 in v.items()}
            for k, v in shadow_case["random_unitary_ids"].items()
        },
        shadow_case["counts"],
    )


shadow_cases_multi = [
    (shadow_case_tmp["answer_multi_shots"], *unpacked_shadow_case(shadow_case_tmp))
    for shadow_case_tmp in raw_shadow_cases
    if "answer_multi_shots" in shadow_case_tmp
]
shadow_cases_single = [
    (shadow_case_tmp["answer_single_shots"], *unpacked_shadow_case(shadow_case_tmp))
    for shadow_case_tmp in raw_shadow_cases
    if "answer_single_shots" in shadow_case_tmp
]
shadow_cases_bitwise = [
    (shadow_case_tmp["answer_bitwise"], *unpacked_shadow_case(shadow_case_tmp))
    for shadow_case_tmp in raw_shadow_cases
    if "answer_bitwise" in shadow_case_tmp
]


def classical_shadow_complex_wrapper(
    arguments: ShadowCaseArguments,
    random_unitary_ids: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
    rho_method: RhoMethodType,
    trace_method: TraceMethodType,
    final_mapping: dict[int, int],
) -> ClassicalShadowComplexExtended:
    """Wrapper for the classical_shadow_complex function to include the trace of the expect_rho.

    Args:
        arguments (ShadowCaseArguments): The arguments for the shadow case.
        random_unitary_ids (dict[int, dict[int, int]]): The random unitary IDs.
        counts (list[dict[str, int]]): The counts.
        rho_method (RhoMethodType): The Rho method.
        trace_method (TraceMethodType): The trace method.
        final_mapping (dict[int, int]): The final mapping.

    Return:
        ClassicalShadowComplexExtended: the result to compare.
    """
    len_register = len(final_mapping)
    random_basis_array = []
    for i in range(len(random_unitary_ids)):
        tmp = {ci: random_unitary_ids[i][n_u_qi] for n_u_qi, ci in final_mapping.items()}
        random_basis_array.append([tmp[j] for j in range(len_register)])

    tmp = classical_shadow_complex(
        shots=arguments["shots"],
        counts=counts,
        random_basis_array=random_basis_array,
        selected_classical_registers=[final_mapping[qi] for qi in arguments["selected_qubits"]],
        rho_method=rho_method,
        trace_method=trace_method,
    )
    return {
        "mean_of_rho_trace": np.trace(tmp["mean_of_rho"]),
        **tmp,
    }


def comparison_shadow_result(
    results: dict[str, ClassicalShadowComplexExtended], answer: dict[str, int]
):
    """Compare the results of different methods.

    Args:
        results (dict[str, ClassicalShadowComplexExtended]): The results to compare.
        answer (dict[str, int]): The expected answer.
    """
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


def test_availability():
    """Test the availability of the Rust backend for the entangled_entropy_core function."""

    for module_location, avails_backends, errors in [  # type: ignore
        classical_shadow_rho_process_availability,
        classical_shadow_matrix_availability,
    ]:
        avails_backends: dict[str, str]
        for backend, status in avails_backends.items():
            assert status, (
                f"{backend} is not available in {module_location}. "
                + f"Check the error: {errors.get(backend)}."
            )


@pytest.mark.parametrize(
    ["answer", "arguments", "random_unitary_ids", "counts"], shadow_cases_multi
)
def test_shadow_multi(
    answer: dict[str, int],
    arguments: ShadowCaseArguments,
    random_unitary_ids: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
):
    """Test the classical_shadow_complex function."""

    _bitstring_mapping, final_mapping = bitstring_mapping_getter(
        counts, arguments["registers_mapping"]
    )

    results = {
        f"{rho_method}.{trace_method}": classical_shadow_complex_wrapper(
            arguments,
            random_unitary_ids,
            counts,
            rho_method,
            trace_method,
            final_mapping,
        )
        for rho_method, trace_method in methods_by_kind["multi_shots"]
    }
    comparison_shadow_result(results, answer)


@pytest.mark.parametrize(
    ["answer", "arguments", "random_unitary_ids", "counts"], shadow_cases_single
)
def test_shadow_single(
    answer: dict[str, int],
    arguments: ShadowCaseArguments,
    random_unitary_ids: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
):
    """Test the classical_shadow_complex function."""

    _bitstring_mapping, final_mapping = bitstring_mapping_getter(
        counts, arguments["registers_mapping"]
    )

    results_spreadout = {
        f"{rho_method}.{trace_method}": classical_shadow_complex_wrapper(
            arguments,
            random_unitary_ids,
            counts,
            rho_method,
            trace_method,
            final_mapping,
        )
        for rho_method, trace_method in methods_by_kind["single_shots"]
    }
    comparison_shadow_result(results_spreadout, answer)


@pytest.mark.parametrize(
    ["answer", "arguments", "random_unitary_ids", "counts"], shadow_cases_bitwise
)
def test_shadow_bitwise(
    answer: dict[str, int],
    arguments: ShadowCaseArguments,
    random_unitary_ids: dict[int, dict[int, int]],
    counts: list[dict[str, int]],
):
    """Test the classical_shadow_complex function."""

    _bitstring_mapping, final_mapping = bitstring_mapping_getter(
        counts, arguments["registers_mapping"]
    )

    results_spreadout = {
        f"{rho_method}.{trace_method}": classical_shadow_complex_wrapper(
            arguments,
            random_unitary_ids,
            counts,
            rho_method,
            trace_method,
            final_mapping,
        )
        for rho_method, trace_method in methods_by_kind["bitwise"]
    }
    comparison_shadow_result(results_spreadout, answer)
