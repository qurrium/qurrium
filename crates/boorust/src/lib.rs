mod bit_slice;
mod counts_process;
mod hadamard;
mod magnet_square;
mod randomized;
mod shadow;
mod string_operator;
mod tool;

use pyo3::prelude::*;

use crate::bit_slice::{cycling_slice_rust, degree_handler_rust, qubit_selector_rust};
use crate::counts_process::{
    counts_list_recount_rust, counts_list_vectorize_rust, rho_m_flatten_counts_list_vectorize_rust,
    shot_counts_selected_clreg_checker, single_counts_recount_rust,
};
use crate::hadamard::purity_echo_core_rust;
use crate::magnet_square::{magnet_square_core_rust, z_dir_magnet_square_core_rust};
use crate::randomized::echo::v1::overlap_echo_core_rust;
use crate::randomized::echo::v2::overlap_echo_core_2_rust;
use crate::randomized::entropy::v1::entangled_entropy_core_rust;
use crate::randomized::entropy::v2::entangled_entropy_core_2_rust;
use crate::randomized::randomized::{ensemble_cell_rust, hamming_distance_rust};
use crate::shadow::nomatmul_trace::nomatmul_trace_sum_rust;
use crate::string_operator::string_operator_core_rust;
use crate::tool::{make_dummy_case_32, make_two_bit_str_32, make_two_bit_str_unlimit};

#[pymodule]
fn boorust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_child_module(m)?;
    Ok(())
}

fn register_child_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let randomized = PyModule::new(parent_module.py(), "randomized")?;
    // construct
    randomized.add_function(wrap_pyfunction!(ensemble_cell_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(hamming_distance_rust, &randomized)?)?;
    // core
    randomized.add_function(wrap_pyfunction!(entangled_entropy_core_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(
        entangled_entropy_core_2_rust,
        &randomized
    )?)?;
    randomized.add_function(wrap_pyfunction!(overlap_echo_core_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(overlap_echo_core_2_rust, &randomized)?)?;

    let counts_process = PyModule::new(parent_module.py(), "counts_process")?;
    counts_process.add_function(wrap_pyfunction!(
        single_counts_recount_rust,
        &counts_process
    )?)?;
    counts_process.add_function(wrap_pyfunction!(counts_list_recount_rust, &counts_process)?)?;
    counts_process.add_function(wrap_pyfunction!(
        shot_counts_selected_clreg_checker,
        &counts_process
    )?)?;
    counts_process.add_function(wrap_pyfunction!(
        counts_list_vectorize_rust,
        &counts_process
    )?)?;
    counts_process.add_function(wrap_pyfunction!(
        rho_m_flatten_counts_list_vectorize_rust,
        &counts_process
    )?)?;

    let bit_slice = PyModule::new(parent_module.py(), "bit_slice")?;
    bit_slice.add_function(wrap_pyfunction!(qubit_selector_rust, &bit_slice)?)?;
    bit_slice.add_function(wrap_pyfunction!(cycling_slice_rust, &bit_slice)?)?;
    bit_slice.add_function(wrap_pyfunction!(degree_handler_rust, &bit_slice)?)?;

    let hadamard = PyModule::new(parent_module.py(), "hadamard")?;
    hadamard.add_function(wrap_pyfunction!(purity_echo_core_rust, &hadamard)?)?;

    let magnet_square = PyModule::new(parent_module.py(), "magnet_square")?;
    magnet_square.add_function(wrap_pyfunction!(magnet_square_core_rust, &magnet_square)?)?;
    magnet_square.add_function(wrap_pyfunction!(
        z_dir_magnet_square_core_rust,
        &magnet_square
    )?)?;

    let string_operator = PyModule::new(parent_module.py(), "string_operator")?;
    string_operator.add_function(wrap_pyfunction!(
        string_operator_core_rust,
        &string_operator
    )?)?;

    let shadow = PyModule::new(parent_module.py(), "shadow")?;
    shadow.add_function(wrap_pyfunction!(nomatmul_trace_sum_rust, &shadow)?)?;

    let dummy = PyModule::new(parent_module.py(), "dummy")?;
    dummy.add_function(wrap_pyfunction!(make_two_bit_str_32, &dummy)?)?;
    dummy.add_function(wrap_pyfunction!(make_dummy_case_32, &dummy)?)?;
    dummy.add_function(wrap_pyfunction!(make_two_bit_str_unlimit, &dummy)?)?;

    let test = PyModule::new(parent_module.py(), "test")?;
    // Null module for now, can add test functions later if needed

    parent_module.add("randomized", &randomized)?;
    parent_module.add_submodule(&randomized)?;
    parent_module.add("counts_process", &counts_process)?;
    parent_module.add_submodule(&counts_process)?;
    parent_module.add("bit_slice", &bit_slice)?;
    parent_module.add_submodule(&bit_slice)?;
    parent_module.add("hadamard", &hadamard)?;
    parent_module.add_submodule(&hadamard)?;
    parent_module.add("magnet_square", &magnet_square)?;
    parent_module.add_submodule(&magnet_square)?;
    parent_module.add("string_operator", &string_operator)?;
    parent_module.add_submodule(&string_operator)?;
    parent_module.add("shadow", &shadow)?;
    parent_module.add_submodule(&shadow)?;
    parent_module.add("dummy", &dummy)?;
    parent_module.add_submodule(&dummy)?;
    parent_module.add("test", &test)?;
    parent_module.add_submodule(&test)?;
    Ok(())
}

// """
// In PyO3 >= 0.23, add_submodule no longer adds the submodule as an attribute
// of the parent module (it only registers it in sys.modules).
// We therefore call parent_module.add("name", &submodule) explicitly so that
// attribute access (e.g. boorust.counts_process) continues to work.
// The Python __init__.py relies on this attribute access to populate sys.modules
// with the fully-qualified names (e.g. "qurry.boorust.counts_process").
// See https://github.com/PyO3/pyo3/issues/1517 for background.
// """
