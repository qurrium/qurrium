use pyo3::prelude::*;
use std::collections::HashMap;
use std::panic;

pub fn check_invalid_counts(shots: i32, counts: &Vec<HashMap<String, i32>>) {
    let invalid_counts = counts
        .iter()
        .enumerate()
        .filter(|(_i, single_counts)| {
            let sample_shots: i32 = single_counts.values().sum();
            shots != sample_shots
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if !invalid_counts.is_empty() {
        panic!(
            "The counts must be equal to the number of shots, but following counts are invalid, index: {:?}",
            invalid_counts
        );
    }
}

pub fn single_counts_recount_prototype(
    single_counts: &HashMap<String, i32>,
    num_classical_registers: i32,
    select_clregs_sort_rev: &Vec<i32>,
) -> HashMap<String, i32> {
    let mut single_counts_recounted: HashMap<String, i32> = HashMap::new();
    for (bit_string_all, count) in single_counts {
        let substring = select_clregs_sort_rev
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
    let mut select_clregs_sort_rev = selected_classical_registers;
    select_clregs_sort_rev.sort_by(|a, b| b.cmp(a));

    single_counts_recount_prototype(
        &single_counts,
        num_classical_registers,
        &select_clregs_sort_rev,
    )
}

#[pyfunction]
#[pyo3(signature = (counts, num_classical_registers, selected_classical_registers))]
pub fn counts_list_recount_rust(
    counts: Vec<HashMap<String, i32>>,
    num_classical_registers: i32,
    selected_classical_registers: Vec<i32>,
) -> Vec<HashMap<String, i32>> {
    let mut select_clregs_sort_rev = selected_classical_registers;
    select_clregs_sort_rev.sort_by(|a, b| b.cmp(a));

    let counts_list_recounted = counts
        .iter()
        .map(|single_counts| {
            single_counts_recount_prototype(
                single_counts,
                num_classical_registers,
                &select_clregs_sort_rev,
            )
        })
        .collect();
    counts_list_recounted
}

pub fn shot_counts_selected_clreg_checker_prototype(
    shots: i32,
    counts: &Vec<HashMap<String, i32>>,
    selected_classical_registers: Option<Vec<i32>>,
) -> (i32, Vec<i32>) {
    // check if the sum of shots is equal to the sum of all counts
    check_invalid_counts(shots, counts);

    // Determine the size of the allsystems
    let total_system_size: i32 = counts[0].keys().next().unwrap().len() as i32;

    let selected_classical_registers_actual = match selected_classical_registers {
        Some(selected_classical_registers) => selected_classical_registers,
        None => (0..total_system_size).collect(),
    };
    for q_i in selected_classical_registers_actual.iter() {
        assert!(
            *q_i >= 0 && *q_i < total_system_size,
            "Invalid selected classical registers: {:?}",
            selected_classical_registers_actual
        );
    }

    (total_system_size, selected_classical_registers_actual)
}

#[pyfunction]
#[pyo3(signature = (shots, counts, selected_classical_registers = None))]
pub fn shot_counts_selected_clreg_checker(
    shots: i32,
    counts: Vec<HashMap<String, i32>>,
    selected_classical_registers: Option<Vec<i32>>,
) -> (i32, Vec<i32>) {
    shot_counts_selected_clreg_checker_prototype(shots, &counts, selected_classical_registers)
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

fn process_vectorize_single_counts(
    single_counts: &HashMap<String, i32>,
    um_data: &Vec<i32>,
    selected_cregs_sorted: &Vec<i32>,
    n_qubits: usize,
) -> (Vec<Vec<i32>>, Vec<i32>) {
    let mut bitstrings = Vec::with_capacity(single_counts.len());
    let mut counts_vec = Vec::with_capacity(single_counts.len());

    for (bit_string, count) in single_counts {
        assert_eq!(
            bit_string.len(),
            n_qubits,
            "bit_string length {} does not match selected_classical_registers_sorted length {}",
            bit_string.len(),
            n_qubits
        );

        let bitstring_vec = bit_string
            .as_bytes()
            .iter()
            .enumerate()
            .map(|(q_idx, &byte)| {
                let direction = um_data[selected_cregs_sorted[q_idx] as usize];
                let digit = (byte - b'0') as i32;
                direction * 10 + digit
            })
            .collect::<Vec<i32>>();

        bitstrings.push(bitstring_vec);
        counts_vec.push(*count);
    }

    (bitstrings, counts_vec)
}

#[pyfunction]
#[pyo3(signature = (counts, random_unitary_array, selected_classical_registers_sorted))]
pub fn rho_m_flatten_counts_list_vectorize_rust(
    counts: Vec<HashMap<String, i32>>,
    random_unitary_array: Vec<Vec<i32>>,
    selected_classical_registers_sorted: Vec<i32>,
) -> Vec<(Vec<Vec<i32>>, Vec<i32>)> {
    let n_qubits = selected_classical_registers_sorted.len();

    counts
        .iter()
        .enumerate()
        .map(|(idx, single_counts)| {
            process_vectorize_single_counts(
                single_counts,
                &random_unitary_array[idx],
                &selected_classical_registers_sorted,
                n_qubits,
            )
        })
        .collect()
}
