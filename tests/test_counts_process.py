"""Tests qurry.process.utils.construct module."""

import os
import pytest

from qurry.capsule import quickRead, quickJSON
from qurry.tools.datetime import current_time
from qurry.process.utils import counts_process_availability
from qurry.process.utils.counts_process import (
    single_counts_recount as single_counts_recount_py,
    single_counts_recount_rust_source,
    counts_list_recount as counts_list_recount_py,
    counts_list_recount_rust_source,
    counts_list_vectorize_pyrust,
    counts_list_vectorize_rust_source,
    rho_m_flatten_counts_list_vectorize_pyrust,
    rho_m_flatten_counts_list_vectorize_rust_source,
)


FILE_LOCATION = os.path.join(os.path.dirname(__file__), "process", "easy-dummy.json")

easy_dummy: dict[str, dict[str, int]] = quickRead(FILE_LOCATION)

test_setup_counts_substring: list[list[int]] = (
    [[i] for i in range(8)]
    + [[i, i + 1] for i in range(7)]
    + [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
)


@pytest.mark.parametrize("test_items", test_setup_counts_substring)
def test_counts_substring(test_items: list[int]):
    """Test the ensemble_cell_rust function."""

    assert counts_process_availability[1]["Rust"], (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )

    counts_recounted_py_result = single_counts_recount_py(easy_dummy["0"], 8, test_items)
    counts_recounted_rust_result = single_counts_recount_rust_source(easy_dummy["0"], 8, test_items)

    assert all(
        counts_recounted_rust_result[s] == v for s, v in counts_recounted_py_result.items()
    ), (
        "Rust and Python results are not equal in counts_recount. "
        + f"test_items: {test_items}, "
        + f"counts_recount_rust_result: {counts_recounted_rust_result}, "
        + f"counts_recount_py_result: {counts_recounted_py_result}."
    )

    counts_list_recounted_py_result = counts_list_recount_py([easy_dummy["0"]], 8, test_items)
    counts_list_recounted_rust_result = counts_list_recount_rust_source(
        [easy_dummy["0"]], 8, test_items
    )

    assert all(
        counts_list_recounted_rust_result[0][s] == v
        for s, v in counts_list_recounted_py_result[0].items()
    ), (
        "Rust and Python results are not equal in counts_list_recount. "
        + f"test_items: {test_items}, "
        + f"counts_list_recount_rust_result: {counts_list_recounted_rust_result}."
        + f"counts_list_recount_py_result: {counts_list_recounted_py_result}, "
    )


def test_counts_list_vectorize():
    """Test the counts_list_vectorize function."""

    assert counts_process_availability[1]["Rust"], (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )

    origin_counts_list = [easy_dummy["0"]]
    counts_list_vectorize_py_result = counts_list_vectorize_pyrust(
        origin_counts_list, backend="Python"
    )
    counts_list_vectorize_rust_result = counts_list_vectorize_rust_source(origin_counts_list)

    error_log_location = os.path.join(os.path.dirname(__file__), "qurrium", "exports")
    current_time_str = current_time().replace(":", "-").replace(" ", "_")

    error_collect_py = {}
    for idx, ((bit_array, value_array), single_counts) in enumerate(
        zip(counts_list_vectorize_py_result, origin_counts_list)
    ):
        tmp = []
        for bit, v in zip(bit_array, value_array):
            bitstring_recover = "".join([str(b) for b in bit])
            if v != single_counts[bitstring_recover]:
                tmp.append((bitstring_recover, v, single_counts[bitstring_recover]))
        if tmp:
            error_collect_py[idx] = tmp

    if error_collect_py:
        quickJSON(
            error_collect_py,
            mode="w+",
            filename=f"error_collect_py.{current_time_str}.json",
            save_location=error_log_location,
        )
    assert not error_collect_py, (
        "Python results are not equal in counts_list_vectorize. "
        + f"See the error log at {error_log_location}, "
        + f"filename: error_collect_py.{current_time_str}.json, "
    )

    error_collect_rust = {}
    for idx, ((bit_array, value_array), single_counts) in enumerate(
        zip(counts_list_vectorize_rust_result, origin_counts_list)
    ):
        tmp = []
        for bit, v in zip(bit_array, value_array):
            bitstring_recover = "".join([str(b) for b in bit])
            if v != single_counts[bitstring_recover]:
                tmp.append((bitstring_recover, v, single_counts[bitstring_recover]))
        if tmp:
            error_collect_rust[idx] = tmp

    if error_collect_rust:
        quickJSON(
            error_collect_rust,
            mode="w+",
            filename=f"error_collect_rust.{current_time_str}.json",
            save_location=error_log_location,
        )
    assert not error_collect_rust, (
        "Rust results are not equal in counts_list_vectorize. "
        + f"See the error log at {error_log_location}, "
        + f"filename: error_collect_rust.{current_time_str}.json, "
    )


def test_rho_m_flatten_counts_list_vectorize():
    """Test the rho_m_flatten_counts_list_vectorize function."""

    assert counts_process_availability[1]["Rust"], (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )

    origin_counts_list = [easy_dummy["0"]]
    rho_m_flatten_counts_list_vectorize_py_result = rho_m_flatten_counts_list_vectorize_pyrust(
        origin_counts_list, {0: dict.fromkeys(range(8), 0)}, list(range(8)), backend="Python"
    )
    rho_m_flatten_counts_list_vectorize_rust_result = (
        rho_m_flatten_counts_list_vectorize_rust_source(
            origin_counts_list, {0: dict.fromkeys(range(8), 0)}, list(range(8))
        )
    )

    error_log_location = os.path.join(os.path.dirname(__file__), "qurrium", "exports")
    current_time_str = current_time().replace(":", "-").replace(" ", "_")

    error_collect_py = {}
    for idx, ((bit_array, value_array), single_counts) in enumerate(
        zip(rho_m_flatten_counts_list_vectorize_py_result, origin_counts_list)
    ):
        tmp = []
        for bit, v in zip(bit_array, value_array):
            bitstring_recover = "".join([str(b) for b in bit])
            if v != single_counts[bitstring_recover]:
                tmp.append((bitstring_recover, v, single_counts[bitstring_recover]))
        if tmp:
            error_collect_py[idx] = tmp

    if error_collect_py:
        quickJSON(
            error_collect_py,
            mode="w+",
            filename=f"error_collect_py.{current_time_str}.json",
            save_location=error_log_location,
        )
    assert not error_collect_py, (
        "Python results are not equal in counts_list_vectorize. "
        + f"See the error log at {error_log_location}, "
        + f"filename: error_collect_py.{current_time_str}.json, "
    )

    error_collect_rust = {}
    for idx, ((bit_array, value_array), single_counts) in enumerate(
        zip(rho_m_flatten_counts_list_vectorize_rust_result, origin_counts_list)
    ):
        tmp = []
        for bit, v in zip(bit_array, value_array):
            bitstring_recover = "".join([str(b) for b in bit])
            if v != single_counts[bitstring_recover]:
                tmp.append((bitstring_recover, v, single_counts[bitstring_recover]))
        if tmp:
            error_collect_rust[idx] = tmp

    if error_collect_rust:
        quickJSON(
            error_collect_rust,
            mode="w+",
            filename=f"error_collect_rust.{current_time_str}.json",
            save_location=error_log_location,
        )
    assert not error_collect_rust, (
        "Rust results are not equal in counts_list_vectorize. "
        + f"See the error log at {error_log_location}, "
        + f"filename: error_collect_rust.{current_time_str}.json, "
    )
