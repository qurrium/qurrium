"""Qurrium Post Processing Exceptions (:mod:`qurry.process.exceptions`)"""


class QurryPostProcessingError(Exception):
    """Base class for errors raised by Qurry."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class PostProcessingCythonImportError(QurryPostProcessingError, ImportError):
    """Cython import error."""


class PostProcessingRustImportError(QurryPostProcessingError, ImportError):
    """Rust import error."""


class PostProcessingThirdPartyImportError(QurryPostProcessingError, ImportError):
    """Third party import error."""


class ClassicalShadowError(QurryPostProcessingError):
    """Base class for errors raised by Classical Shadow post-processing."""


class AccuracyProbabilityCalculationError(ClassicalShadowError, ValueError):
    """Invalid accuracy probability component delta for Classical Shadow post-processing."""


# General Warning
class QurryPostProcessingWarning(Warning):
    """Base class for warning raised by Qurry."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class PostProcessingCythonUnavailableWarning(QurryPostProcessingWarning):
    """Cython unavailable warning."""


class PostProcessingRustUnavailableWarning(QurryPostProcessingWarning):
    """Rust unavailable warning."""


class PostProcessingThirdPartyUnavailableWarning(QurryPostProcessingWarning):
    """Third party unavailable warning."""


class PostProcessingBackendDeprecatedWarning(QurryPostProcessingWarning, DeprecationWarning):
    """Post-processing backend is deprecated."""


class ClassicalShadowWarning(QurryPostProcessingWarning):
    """Base class for warning raised by Classical Shadow post-processing."""


class AccuracyProbabilityWarning(ClassicalShadowWarning):
    """Warning for invalid accuracy probability component delta in
    Classical Shadow post-processing."""
