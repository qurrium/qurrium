mod bit_slice;
mod counts_process;
mod hadamard;
mod randomized;
mod tool;

use pyo3::prelude::*;

use crate::bit_slice::{
    cycling_slice_rust, degree_handler_rust, qubit_selector_rust, test_bit_slice,
};
use crate::counts_process::{
    counts_list_recount_rust, counts_list_vectorize_rust, rho_m_flatten_counts_list_vectorize_rust,
    shot_counts_selected_clreg_checker, single_counts_recount_rust,
};
use crate::hadamard::purity_echo_core_rust;
use crate::randomized::echo::v1::{echo_cell_rust, overlap_echo_core_rust};
use crate::randomized::echo::v2::{echo_cell_2_rust, overlap_echo_core_2_rust};
use crate::randomized::entropy::v1::{entangled_entropy_core_rust, purity_cell_rust};
use crate::randomized::entropy::v2::{entangled_entropy_core_2_rust, purity_cell_2_rust};
use crate::randomized::randomized::{ensemble_cell_rust, hamming_distance_rust};
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
    randomized.add_function(wrap_pyfunction!(purity_cell_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(echo_cell_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(purity_cell_2_rust, &randomized)?)?;
    randomized.add_function(wrap_pyfunction!(echo_cell_2_rust, &randomized)?)?;
    // main
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

    let dummy = PyModule::new(parent_module.py(), "dummy")?;
    dummy.add_function(wrap_pyfunction!(make_two_bit_str_32, &dummy)?)?;
    dummy.add_function(wrap_pyfunction!(make_dummy_case_32, &dummy)?)?;
    dummy.add_function(wrap_pyfunction!(make_two_bit_str_unlimit, &dummy)?)?;

    let test = PyModule::new(parent_module.py(), "test")?;
    test.add_function(wrap_pyfunction!(test_bit_slice, &test)?)?;

    parent_module.add_submodule(&randomized)?;
    parent_module.add_submodule(&counts_process)?;
    parent_module.add_submodule(&bit_slice)?;
    parent_module.add_submodule(&hadamard)?;
    parent_module.add_submodule(&dummy)?;
    parent_module.add_submodule(&test)?;
    Ok(())
}

// """
// Note that this does not define a package,
// so this won’t allow Python code to directly import submodules
// by using from parent_module import child_module.
// For more information,
// see [#759](https://github.com/PyO3/pyo3/issues/759)
// and [#1517](https://github.com/PyO3/pyo3/issues/1517).
// from https://pyo3.rs/v0.23.0/module.html#python-submodules
// (Since PyO3 0.20.0, until PyO3 0.23.0)
// :smile:
// """
