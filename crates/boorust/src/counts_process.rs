use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::panic;

pub fn single_counts_under_degree_prototype(
    single_counts: HashMap<String, i32>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> HashMap<String, i32> {
    let mut single_counts_under_degree: HashMap<String, i32> = HashMap::new();
    for (bit_string_all, count) in single_counts {
        let substring = selected_classical_registers
            .iter()
            .map(|&i| {
                bit_string_all
                    .chars()
                    .nth((num_classical_registers - i - 1) as usize)
                    .unwrap_or_else(|| {
                        panic!(
                            "Index out of bounds: num_classical_registers = {}, i = {}, bit_string_all = {}, (num_classical_registers - i - 1) = {}",
                            num_classical_registers,
                            i,
                            bit_string_all,
                            num_classical_registers - i - 1
                        )
                    })
            })
            .collect::<String>();
        let entry = single_counts_under_degree
            .entry(substring.to_string())
            .or_insert(0);
        *entry += count;
    }
    single_counts_under_degree
}

#[pyfunction]
#[pyo3(signature = (single_counts, num_classical_registers, selected_classical_registers))]
pub fn single_counts_under_degree_rust(
    single_counts: HashMap<String, i32>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> HashMap<String, i32> {
    single_counts_under_degree_prototype(
        single_counts,
        num_classical_registers,
        selected_classical_registers,
    )
}

#[pyfunction]
#[pyo3(signature = (counts, num_classical_registers, selected_classical_registers))]
pub fn counts_list_under_degree_rust(
    counts: Vec<HashMap<String, i32>>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> Vec<HashMap<String, i32>> {
    let mut counts_list_under_degree: Vec<HashMap<String, i32>> = Vec::new();
    for single_counts in counts {
        let counts = single_counts_under_degree_prototype(
            single_counts,
            num_classical_registers,
            selected_classical_registers.clone(),
        );
        counts_list_under_degree.push(counts);
    }
    counts_list_under_degree
}

#[pyfunction]
#[pyo3(signature = (counts))]
pub fn counts_list_vectorize_rust(
    counts: Vec<HashMap<String, i32>>,
) -> Vec<(Vec<Vec<i32>>, Vec<i32>)> {
    let mut counts_list_vectorized: Vec<(Vec<Vec<i32>>, Vec<i32>)> = Vec::new();
    for single_counts in counts {
        let mut bitstrings: Vec<Vec<i32>> = Vec::new();
        let mut counts_vec: Vec<i32> = Vec::new();
        for (bit_string, count) in single_counts {
            let bitstring_vec: Vec<i32> = bit_string
                .chars()
                .map(|c| c.to_digit(2).unwrap() as i32)
                .collect();
            bitstrings.push(bitstring_vec);
            counts_vec.push(count);
        }
        counts_list_vectorized.push((bitstrings, counts_vec));
    }
    counts_list_vectorized
}

#[pyfunction]
#[pyo3(signature = (shots, counts, selected_classical_registers = None))]
pub fn shot_counts_selected_clreg_checker(
    shots: i32,
    counts: Vec<HashMap<String, i32>>,
    selected_classical_registers: Option<Vec<i32>>,
) -> (i32, Vec<i32>) {
    // check if the sum of shots is equal to the sum of all counts
    let sample_shots: i32 = counts[0].values().sum();
    assert_eq!(
        shots, sample_shots,
        "shots {} does not match sample_shots {}",
        shots, sample_shots
    );

    // Determine the size of the allsystems
    let measured_system_size: i32 = counts[0].keys().next().unwrap().len() as i32;

    let selected_classical_registers_actual = match selected_classical_registers {
        Some(selected_classical_registers) => selected_classical_registers,
        None => (0..measured_system_size).collect(),
    };
    for q_i in selected_classical_registers_actual.iter() {
        assert!(
            *q_i >= 0 && *q_i < measured_system_size,
            "Invalid selected classical registers: {:?}",
            selected_classical_registers_actual
        );
    }

    (measured_system_size, selected_classical_registers_actual)
}
