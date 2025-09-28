use pyo3::prelude::*;
use rayon::prelude::*;

fn rho_elt_process(
    rho_a_pauli: &usize,
    rho_b_pauli: &usize,
    rho_a_spin: &i32,
    rho_b_spin: &i32,
) -> f64 {
    if rho_a_pauli != rho_b_pauli {
        0.5
    } else {
        if rho_a_spin == rho_b_spin {
            5.0
        } else {
            -4.0
        }
    }
}

fn get_trace(
    rho_a_pauli: &Vec<usize>,
    rho_a_spin: &Vec<i32>,
    rho_b_pauli: &Vec<usize>,
    rho_b_spin: &Vec<i32>,
    subsystem: &Vec<usize>,
) -> f64 {
    if subsystem.is_empty() {
        return 1.0;
    }

    subsystem.iter().fold(1.0, |acc, &i| {
        acc * rho_elt_process(
            &rho_a_pauli[i],
            &rho_b_pauli[i],
            &rho_a_spin[i],
            &rho_b_spin[i],
        )
    })
}

#[pyfunction]
pub fn nomatmul_trace_sum_rust(
    pauli_basis: Vec<Vec<usize>>,
    spin_outcome: Vec<Vec<i32>>,
    subsystem: Vec<usize>,
) -> f64 {
    let num_samples = pauli_basis.len();

    let result = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let row_sum: f64 = (i + 1..num_samples)
                .into_par_iter()
                .map(|j| {
                    get_trace(
                        &pauli_basis[i],
                        &spin_outcome[i],
                        &pauli_basis[j],
                        &spin_outcome[j],
                        &subsystem,
                    )
                })
                .sum();
            row_sum
        })
        .sum();

    result
}
