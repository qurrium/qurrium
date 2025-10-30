"""Post Processing - Classical Shadow - Trace Process - Non-Matrix Operation Core
(:mod:`qurry.process.classical_shadow.trace_process.nomatop_core`)

"""

from typing import Iterable, Literal, Union, Optional

from .nomatmul_trace import nomatmul_trace_core, NonMatMulTraceMethod
from .bitwise import bitwise_core, BitWiseTraceMethod
from ..utils import multi_counts_to_basis_spin
from ...utils import shot_counts_selected_clreg_checker_pyrust, BaseMethodEnum


class NonMatOpTraceMethod(BaseMethodEnum):
    """TThe method to use for the trace calculation without matrix multiplication.

    - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
    - "nomatmul_trace_rust": Use Rust implementation via PyO3.
    - "bitwise_py": Use pure Python bitwise implementation.

    The default method is "bitwise_py", which is the fastest option.
    """

    NOMATMUL_TRACE_PY = NonMatMulTraceMethod.NOMATMUL_TRACE_PY.value
    """Use pure Python implementation without multiprocessing."""

    # NOMATMUL_TRACE_PY_MP = NonMatMulTraceMethod.NOMATMUL_TRACE_PY_MP.value
    # """Use pure Python implementation with multiprocessing."""

    NOMATMUL_TRACE_RUST = NonMatMulTraceMethod.NOMATMUL_TRACE_RUST.value
    """Use Rust implementation via PyO3."""

    BITWISE_PY = BitWiseTraceMethod.BITWISE_PY.value
    """Use pure Python bitwise implementation."""

    @classmethod
    def get_default(cls) -> "NonMatOpTraceMethod":
        """Get the default method.

        Returns:
            The default method.
        """
        return cls.BITWISE_PY

    def is_nomatmul_method(self) -> bool:
        """Whether it is a nomatmul method.

        Returns:
            bool: True if it is a nomatmul method, False otherwise.
        """
        return self in [self.NOMATMUL_TRACE_PY, self.NOMATMUL_TRACE_RUST]

    def is_bitwise_method(self) -> bool:
        """Whether it is a bitwise method.

        Returns:
            bool: True if it is a bitwise method, False otherwise.
        """
        return self in [self.BITWISE_PY]

    def to_nomatmul_enum(self) -> NonMatMulTraceMethod:
        """Convert to NonMatMulTraceMethod enum.

        Raises:
            ValueError: If the method is not a nomatmul method.

        Returns:
            NonMatMulTraceMethod: The corresponding NonMatMulTraceMethod enum.
        """
        if not self.is_nomatmul_method():
            raise ValueError(f"{self.value} is not a nomatmul method")
        return NonMatMulTraceMethod.from_string(self.value)

    def to_bitwise_enum(self) -> BitWiseTraceMethod:
        """Convert to BitWiseTraceMethod enum.

        Raises:
            ValueError: If the method is not a bitwise method.

        Returns:
            BitWiseTraceMethod: The corresponding BitWiseTraceMethod enum.
        """
        if not self.is_bitwise_method():
            raise ValueError(f"{self.value} is not a bitwise method")
        return BitWiseTraceMethod.from_string(self.value)


NonMatOpTraceMethodType = Union[NonMatOpTraceMethod, str]
"""The method to use for the trace calculation without matrix multiplication.

- "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
- "nomatmul_trace_rust": Use Rust implementation via PyO3.
- "bitwise_py": Use pure Python bitwise implementation.

The default method is "bitwise_py", which is the fastest option.
"""

DEFAULT_NONMATOP_TRACE_METHOD: NonMatOpTraceMethod = NonMatOpTraceMethod.get_default()
"""The default method for the trace calculation without matrix multiplication."""


def trace_nomatop_core(
    shots: int,
    counts: list[dict[str, int]],
    random_unitary_array: list[list[Union[Literal[0, 1, 2], int]]],
    selected_classical_registers: Optional[Iterable[int]] = None,
    trace_method: NonMatOpTraceMethodType = DEFAULT_NONMATOP_TRACE_METHOD,
) -> float:
    """Calculate the trace using non-matrix operation methods.

    Args:
        shots (int):
            The number of shots.
        counts (list[dict[str, int]]):
            The list of the counts.
        random_basis_array (list[list[Union[Literal[0, 1, 2], int]]],):
            The shadow direction of the unitary operators.
        selected_classical_registers (Optional[Iterable[int]], optional):
            The list of **the index of the selected_classical_registers**.
            Defaults to None.
        trace_method (NonMatOpTraceMethodType, optional):
            The method to use for the trace calculation without matrix multiplication.

            - "nomatmul_trace_py": Use pure Python implementation without multiprocessing.
            - "nomatmul_trace_rust": Use Rust implementation via PyO3.
            - "bitwise_py": Use pure Python bitwise implementation.

            Currently, the "bitwise_py" method is the fastest.
            Defaults to DEFAULT_NONMATOP_TRACE_METHOD, which is "bitwise_py".

    Returns:
        float: The purity.
    """

    _total_system_size, selected_classical_registers = shot_counts_selected_clreg_checker_pyrust(
        shots=shots,
        counts=counts,
        selected_classical_registers=selected_classical_registers,
    )
    selected_clreg_sorted = sorted(selected_classical_registers)
    pauli_basis, spin_outcome = multi_counts_to_basis_spin(shots, counts, random_unitary_array)

    if isinstance(trace_method, str):
        trace_method = NonMatOpTraceMethod.from_string(trace_method)

    if trace_method.is_bitwise_method():
        return bitwise_core(pauli_basis, spin_outcome, selected_clreg_sorted)
    if trace_method.is_nomatmul_method():
        nomatmul_enum = trace_method.to_nomatmul_enum()
        return nomatmul_trace_core(
            pauli_basis, spin_outcome, selected_clreg_sorted, trace_method=nomatmul_enum
        )

    raise trace_method.value_error()
