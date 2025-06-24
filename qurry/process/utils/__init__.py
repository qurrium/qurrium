"""Utility functions for qurry.process (:mod:`qurry.process.utils`)"""

from .counts_process import (
    single_counts_recount,
    single_counts_recount_pyrust,
    counts_list_recount,
    counts_list_recount_pyrust,
    BACKEND_AVAILABLE as counts_process_availability,
    shot_counts_selected_clreg_checker_pyrust,
    counts_list_vectorize_pyrust,
    rho_m_flatten_counts_list_vectorize_pyrust,
)
from .other import NUMERICAL_ERROR_TOLERANCE
from .bit_slice import (
    qubit_selector,
    cycling_slice,
    degree_handler,
    qubit_mapper,
    is_cycling_slice_active,
    BACKEND_AVAILABLE as bit_slice_availability,
)
from .randomized import (
    hamming_distance,
    ensemble_cell,
    BACKEND_AVAILABLE as randomized_availability,
)
from .dummy import BACKEND_AVAILABLE as dummy_availability
from .test import BACKEND_AVAILABLE as test_availability, test_bit_slice
