"""Tests qurry.process.utils.construct module."""

import os
import pytest

from qurry.capsule import quick_json_write
from qurry.tools.datetime import current_time
from qurry.process.utils import counts_process_availability
from qurry.process.utils.counts_process import (
    single_counts_recount_pyrust,
    counts_list_recount_pyrust,
    counts_list_vectorize_pyrust,
    counts_list_vectorize_rust,
    rho_m_flatten_counts_list_vectorize_pyrust,
    rho_m_flatten_counts_list_vectorize_rust,
)

from utilities import quick_json_read, get_dummy_file_path


EASY_DUMMY_PATH = get_dummy_file_path("easy_dummy.json")
easy_dummy_raw: dict[str, dict[str, int]] = quick_json_read(EASY_DUMMY_PATH)
easy_dummy: dict[int, dict[str, int]] = {int(k): v for k, v in easy_dummy_raw.items()}

ERROR_LOG_LOCATION = os.path.join(os.path.dirname(__file__), "..", "qurrium", "exports")


def make_current_time_str() -> str:
    """Make the current time string for error log filenames."""
    return current_time().replace(":", "-").replace(" ", "_")


def assert_error_log_json(error_collect: dict, current_time_str: str, msg: str):
    """Export the error log location for test modules."""
    if error_collect:
        if not os.path.exists(ERROR_LOG_LOCATION):
            os.mkdir(ERROR_LOG_LOCATION)
        quick_json_write(
            error_collect,
            mode="w+",
            filename=f"error_collect.{current_time_str}.json",
            save_location=ERROR_LOG_LOCATION,
        )
    assert not error_collect, (
        msg
        + f"See the error log at {ERROR_LOG_LOCATION}, "
        + f"filename: error_collect_py.{current_time_str}.json, "
    )


test_setup_counts_substring: list[list[int]] = (
    [[i] for i in range(8)]
    + [[i, i + 1] for i in range(7)]
    + [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
)


def test_availability():
    """Test the availability of the Rust backend for the counts_recount function."""

    assert counts_process_availability[1]["Rust"] != "Error", (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )


@pytest.mark.parametrize("test_items", test_setup_counts_substring)
def test_counts_substring(test_items: list[int]):
    """Test the ensemble_cell_rust function."""

    counts_recounted_py_result = single_counts_recount_pyrust(
        easy_dummy[0], 8, test_items, backend="Python"
    )
    counts_recounted_rust_result = single_counts_recount_pyrust(
        easy_dummy[0], 8, test_items, backend="Rust"
    )

    assert all(
        counts_recounted_rust_result[s] == v for s, v in counts_recounted_py_result.items()
    ), (
        "Rust and Python results are not equal in counts_recount. "
        + f"test_items: {test_items}, "
        + f"counts_recount_rust_result: {counts_recounted_rust_result}, "
        + f"counts_recount_py_result: {counts_recounted_py_result}."
    )

    counts_list_recounted_py_result = counts_list_recount_pyrust(
        [easy_dummy[0]], 8, test_items, backend="Python"
    )
    counts_list_recounted_rust_result = counts_list_recount_pyrust(
        [easy_dummy[0]], 8, test_items, backend="Rust"
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

    assert counts_process_availability[1]["Rust"] != "Error", (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )

    origin_counts_list = [easy_dummy[0]]
    counts_list_vectorize_py_result = counts_list_vectorize_pyrust(
        origin_counts_list, backend="Python"
    )
    counts_list_vectorize_rust_result = counts_list_vectorize_rust(origin_counts_list)

    current_time_str = make_current_time_str()

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

    assert_error_log_json(
        error_collect_py,
        current_time_str,
        "Python results are not equal in counts_list_vectorize. ",
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

    assert_error_log_json(
        error_collect_rust,
        current_time_str,
        "Rust results are not equal in counts_list_vectorize. ",
    )


def test_rho_m_flatten_counts_list_vectorize():
    """Test the rho_m_flatten_counts_list_vectorize function."""

    assert counts_process_availability[1]["Rust"] != "Error", (
        "Rust is not available." + f" Check the error: {counts_process_availability[2]}"
    )

    origin_counts_list = [easy_dummy[0]]
    rho_m_flatten_counts_list_vectorize_py_result = rho_m_flatten_counts_list_vectorize_pyrust(
        origin_counts_list, [[0] * 8], list(range(8)), backend="Python"
    )
    rho_m_flatten_counts_list_vectorize_rust_result = rho_m_flatten_counts_list_vectorize_rust(
        origin_counts_list, [[0] * 8], list(range(8))
    )

    current_time_str = make_current_time_str()

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

    assert_error_log_json(
        error_collect_py,
        current_time_str,
        "Python results are not equal in rho_m_flatten_counts_list_vectorize. ",
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

    assert_error_log_json(
        error_collect_rust,
        current_time_str,
        "Rust results are not equal in rho_m_flatten_counts_list_vectorize. ",
    )
