use pyo3::prelude::*;
use std::collections::HashMap;

fn add_or_reducer(bitstring: &str) -> i32 {
    let ones_count = bitstring
        .chars()
        .map(|c| c.to_digit(10).unwrap_or(0)) // 處理 '0' 或 '1'
        .sum::<u32>();

    if ones_count % 2 == 0 {
        1
    } else {
        -1
    }
}

#[pyfunction]
#[pyo3(signature = (shots, counts))]
pub fn string_operator_core_rust(shots: i32, counts: Vec<HashMap<String, i32>>) -> f64 {
    if counts.len() != 1 {
        panic!(
            "counts should be a list of counts with length 1, but got {}",
            counts.len()
        );
    }

    let only_counts = &counts[0];

    let sample_shots: i32 = only_counts.values().sum();
    assert!(
        sample_shots == shots,
        "shots {} does not match sample_shots {}",
        shots,
        sample_shots
    );

    let total: i32 = only_counts
        .iter()
        .map(|(bitstring, &count)| add_or_reducer(bitstring) * count)
        .sum();

    total as f64 / sample_shots as f64
}
