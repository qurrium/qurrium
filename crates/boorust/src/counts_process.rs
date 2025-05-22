use pyo3::prelude::*;
use std::collections::HashMap;
use std::panic;

pub fn single_counts_recount_prototype(
    single_counts: HashMap<String, i32>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> HashMap<String, i32> {
    let mut single_counts_recounted: HashMap<String, i32> = HashMap::new();
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
        let entry = single_counts_recounted
            .entry(substring.to_string())
            .or_insert(0);
        *entry += count;
    }
    single_counts_recounted
}

#[pyfunction]
#[pyo3(signature = (single_counts, num_classical_registers, selected_classical_registers))]
pub fn single_counts_recount_rust(
    single_counts: HashMap<String, i32>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> HashMap<String, i32> {
    single_counts_recount_prototype(
        single_counts,
        num_classical_registers,
        selected_classical_registers,
    )
}

#[pyfunction]
#[pyo3(signature = (counts, num_classical_registers, selected_classical_registers))]
pub fn counts_list_recount_rust(
    counts: Vec<HashMap<String, i32>>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> Vec<HashMap<String, i32>> {
    let mut counts_list_recounted: Vec<HashMap<String, i32>> = Vec::new();
    for single_counts in counts {
        let counts = single_counts_recount_prototype(
            single_counts,
            num_classical_registers,
            selected_classical_registers.clone(),
        );
        counts_list_recounted.push(counts);
    }
    counts_list_recounted
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
#[pyo3(signature = (counts, random_unitary_um, selected_classical_registers_sorted))]
pub fn rho_m_flatten_counts_list_vectorize_rust(
    counts: Vec<HashMap<String, i32>>,
    random_unitary_um: HashMap<i32, HashMap<i32, i32>>,
    selected_classical_registers_sorted: Vec<i32>,
) -> Vec<(Vec<Vec<i32>>, Vec<i32>)> {
    let mut rho_m_flatten_counts_list_vectorized: Vec<(Vec<Vec<i32>>, Vec<i32>)> = Vec::new();

    for (um_idx, single_counts) in counts.iter().enumerate() {
        let mut bitstrings: Vec<Vec<i32>> = Vec::new();
        let mut counts_vec: Vec<i32> = Vec::new();
        for (bit_string, count) in single_counts {
            assert!(
                bit_string.len() == selected_classical_registers_sorted.len(),
                "bit_string length {} does not match selected_classical_registers_sorted length {}",
                bit_string.len(),
                selected_classical_registers_sorted.len()
            );
            let bitstring_vec: Vec<i32> = bit_string
                .chars()
                .enumerate()
                .map(|(q_idx, c)| {
                    random_unitary_um[&(um_idx as i32)][&selected_classical_registers_sorted[q_idx]]
                        * 10
                        + c.to_digit(2).unwrap() as i32
                })
                .collect();
            bitstrings.push(bitstring_vec);
            counts_vec.push(count.clone());
        }
        rho_m_flatten_counts_list_vectorized.push((bitstrings, counts_vec));
    }
    rho_m_flatten_counts_list_vectorized
}
