"""Exceptions (:mod:`qurry.exceptions`)"""


class QurryError(Exception):
    """Base class for errors raised by Qurry."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QurryInvalidInherition(QurryError):
    """Invalid inherition class making by Qurry."""


class QurryExperimentCountsNotCompleted(QurryError):
    """Experiment is not completed."""


class QurryExtraPackageRequired(QurryError, ImportError):
    """Extra package required for Qurry."""


class QurryInvalidArgument(QurryError, ValueError):
    """Invalid argument for Qurry."""


class QurryPositionalArgumentNotSupported(QurryError, ValueError):
    """Positional argument is not supported."""


class QurryCountLost(QurryError):
    """Count lost error."""


class QurryDependenciesFailureError(QurryError):
    """The dependencies of Qurry like Qiskit raise some error."""


class RandomizedMeasureError(QurryError):
    """The error for randomized measure."""


class RandomizedMeasureUnitaryOperatorNotFullCovering(RandomizedMeasureError, ValueError):
    """Randomized measure unitary operator warning
    for not full covering the measure range."""


class OverlapComparisonSizeDifferent(RandomizedMeasureError, ValueError):
    """The sizes between two system that need to be compared are different."""


# General Warning
class QurryWarning(Warning):
    """Base class for warning raised by Qurry."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class UnconfiguredWarning(QurryWarning):
    "For dummy function in qurrium has been activated."


class QurryInheritionNoEffect(QurryWarning):
    "This configuration method has no effect."


class QurryUnrecongnizedArguments(QurryWarning):
    "This argument is not recognized but may be kept at somewhere."


class QurryMemoryOverAllocationWarning(QurryWarning):
    "Automatically shutdown experiment to protect RAM for preventing crashing."


class QurryImportWarning(QurryWarning):
    "Warning for qurry trying to import something."


class QurryResetSecurityActivated(QurryWarning):
    "Warning for reset class security."


class QurryResetAccomplished(QurryWarning):
    "Warning for class reset."


class QurryProtectContent(QurryWarning):
    "Warning for protect content."


class QurrySummonerInfoIncompletion(QurryWarning):
    """Warning for summoner info incompletion.
    The summoner is the instance of :cls:`QurryMultiManager`."""


class QurryDummyRunnerWarning(QurryWarning):
    """Dummy runner warning."""


class QurryHashIDInvalid(QurryWarning):
    """Hash ID invalid warning."""


class QurryArgumentsExpectedNotNone(QurryWarning):
    """Arguments expected not None warning."""


class QurryDependenciesNotWorking(QurryWarning):
    """Some function from the dependencies of Qurry
    like Qiskit will not working for some reason."""


class QurryTranspileConfigurationIgnored(QurryWarning):
    """Transpile configuration ignored warning."""


class QurryPendingTagTooMany(QurryWarning):
    """Pending tag too many warning."""


class QurryDeprecatedWarning(QurryWarning, DeprecationWarning):
    """Deprecated warning."""


class QurryUnknownExportOption(QurryWarning):
    """Unknown export option warning."""


class SeperatedExecutingOverlapResult(QurryWarning):
    """When the seperated executing overlap the result with the same backend"""


class OpenQASMProcessingWarning(QurryWarning):
    """OpenQASM processing warning."""


class OpenQASM3Issue13362Warning(OpenQASMProcessingWarning):
    """OpenQASM3 warning for Qiskit issue 12632.
    You will need to upgrade your Qiskit version to 1.3.2 for fixing this issue.

    - The issues report: https://github.com/Qiskit/qiskit/issues/13362
    - Pull Requests merged:
        1. https://github.com/Qiskit/qiskit/pull/13633
        2. https://github.com/Qiskit/qiskit/pull/13663

    """


class QurryUnprovenFeatureWarning(QurryWarning):
    """Unproven feature warning.
    This feature is not proven to be working or not.
    Please report if you find any issue with this feature.
    """
