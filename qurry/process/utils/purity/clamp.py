"""Post Processing - Utils - Purity - Clamp Purity (:mod:`qurry.process.utils.purity.clamp`)

This method introduced in
`Predicting Properties of Quantum Many-Body Systems
<https://github.com/hsinyuan-huang/predicting-quantum-properties>`_ ,
which can handle the statistical error that cause the purity to be out of the physical range
on both methods of classical shadow and randomized measurement.

"""

from ..other import NUMERICAL_ERROR_TOLERANCE


def clamp_purity(
    purity: float, subsystem_size: int, resolution: float = NUMERICAL_ERROR_TOLERANCE
) -> float:
    """Clamp the purity value to be within the allowed range.

    Args:
        purity (float): The purity value to clamp.
        subsystem_size (int): The size of the subsystem.
        resolution (float): The resolution for clamping. Defaults to NUMERICAL_ERROR_TOLERANCE.

    Returns:
        float: The clamped purity value.
    """
    min_purity = 1.0 / (2.0**subsystem_size)
    max_purity = 1.0 - resolution
    return max(min(purity, max_purity), min_purity)
