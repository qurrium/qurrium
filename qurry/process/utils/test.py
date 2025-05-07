"""Test (:mod:`qurry.process.utils.test`)"""

import warnings

from ..availability import availablility
from ..exceptions import PostProcessingRustImportError, PostProcessingRustUnavailableWarning

try:
    from ...boorust import test  # type: ignore

    test_bit_slice_source = test.test_bit_slice

    RUST_AVAILABLE = True
    FAILED_RUST_IMPORT = None
except ImportError as err:
    RUST_AVAILABLE = False
    FAILED_RUST_IMPORT = err

    def test_bit_slice_source():
        """Dummy function for test_bit_slice."""
        raise PostProcessingRustImportError(
            "Rust is not available, skipping test_bit_slice."
        ) from FAILED_RUST_IMPORT


BACKEND_AVAILABLE = availablility(
    "utils.test",
    [
        ("Rust", RUST_AVAILABLE, FAILED_RUST_IMPORT),
    ],
)


def test_bit_slice():
    """Test the construct module."""

    if RUST_AVAILABLE:
        test_bit_slice_source()
    else:
        warnings.warn(
            f"Rust is not available, Check: {FAILED_RUST_IMPORT}",
            PostProcessingRustUnavailableWarning,
        )
