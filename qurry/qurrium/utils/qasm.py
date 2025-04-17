"""OpenQASM Processor (:mod:`qurry.qurrium.utils.qasm`)"""

from typing import Literal, Optional
import warnings

from qiskit import QuantumCircuit, __version__ as qiskit_version
from qiskit.qasm3 import dumps as dumps_qasm3, QASM3Error, loads as loads_qasm3
from qiskit.qasm2 import dumps as dumps_qasm2, QASM2Error, loads as loads_qasm2

from ...exceptions import OpenQASMProcessingWarning, OpenQASM3Issue13362Warning


def qasm_dumps(
    qc: QuantumCircuit,
    qasm_version: Literal["qasm2", "qasm3"] = "qasm3",
) -> str:
    """Draw the circuits in OpenQASM string.

    Args:
        qc (QuantumCircuit):
            The circuit wanted to be drawn.
        qasm_version (Literal["qasm2", "qasm3"], optional):
            The export version of OpenQASM. Defaults to 'qasm3'.

    Raises:
        ValueError: If the OpenQASM version is invalid.

    Returns:
        str: The drawing of circuit in OpenQASM string.
    """
    if qasm_version == "qasm2":
        try:
            txt = dumps_qasm2(qc)
        except QASM2Error as err:
            txt = f"| Skip dumps into OpenQASM2, due to QASM2Error: {err}"
            # pylint: disable=broad-except
        except Exception as err:
            # pylint: enable=broad-except
            txt = f"| Skip dumps into OpenQASM2, due to critical errors: {err}"
            warnings.warn(
                OpenQASMProcessingWarning(
                    "Critical errors in qiskit.qasm2.dumps, "
                    + f"due to Exception: {err}, give up to export."
                )
            )

    elif qasm_version == "qasm3":
        if tuple(int(v) for v in qiskit_version.split(".")) < (1, 3, 2):
            warnings.warn(
                OpenQASM3Issue13362Warning(
                    "The qiskit version is lower than 1.3.2, "
                    + "which has a critical issue in qiskit.qasm3.dumps. "
                    + "You will need to upgrade your Qiskit "
                    + "version to 1.3.2 for fixing this issue. "
                    + "The issues report: https://github.com/Qiskit/qiskit/issues/13362, "
                    + "Pull Requests merged: "
                    + "1. https://github.com/Qiskit/qiskit/pull/13633, "
                    + "2. https://github.com/Qiskit/qiskit/pull/13663"
                )
            )

        try:
            txt = dumps_qasm3(qc)
        except QASM3Error as err:
            txt = f"| Skip dumps into OpenQASM3, due to QASM3Error: {err}"
            # pylint: disable=broad-except

        except Exception as err:
            # pylint: enable=broad-except
            txt = f"| Skip dumps into OpenQASM3, due to critical errors: {err}"
            if isinstance(err, TypeError) and "only length-1 arrays can be converted" in str(err):
                warnings.warn(
                    OpenQASM3Issue13362Warning(
                        "Critical errors in qiskit.qasm3.dumps, "
                        + f"due to Exception: {err}, give up to export. "
                        + "This issue is caused by a incorrectly implemented function in Qiskit. "
                        + "You will need to upgrade your Qiskit "
                        + "version to 1.3.2 for fixing this issue. "
                        + "The issues report: https://github.com/Qiskit/qiskit/issues/13362, "
                        + "Pull Requests merged: "
                        + "1. https://github.com/Qiskit/qiskit/pull/13633, "
                        + "2. https://github.com/Qiskit/qiskit/pull/13663"
                    )
                )
            else:
                warnings.warn(
                    OpenQASMProcessingWarning(
                        "Critical errors in qiskit.qasm3.dumps, "
                        + f"due to Exception: {err}, give up to export."
                    )
                )
    else:
        raise ValueError(f"Invalid qasm version: {qasm_version}")

    assert isinstance(txt, str), "The drawing of circuit does not export."
    return txt


def qasm_version_detect(qasm_str: str) -> Literal["qasm2", "qasm3"]:
    """Detect the OpenQASM version from the string.

    Args:
        qasm_str (str):
            The OpenQASM string wanted to be detected.

    Returns:
        Literal["qasm2", "qasm3"]: The detected OpenQASM version.
    """
    if "OPENQASM 2.0" in qasm_str:
        return "qasm2"
    if "OPENQASM 3.0" in qasm_str:
        return "qasm3"
    raise ValueError("Invalid OpenQASM version.")


# qasm_loads will be not used anywhere in the project,
# since it will require `qiskit-qasm3-import`
# which has uncertain development status
# that mentioned in https://github.com/Qiskit/qiskit-qasm3-import/issues/13
# It also affects our wanted feature "revive circuit from OpenQASM3".


def qasm_loads(
    qasm_str: str,
    qasm_version: Optional[Literal["qasm2", "qasm3"]] = None,
) -> Optional[QuantumCircuit]:
    """Load the circuits from OpenQASM string.

    Args:
        qasm_str (str):
            The OpenQASM string wanted to be loaded.
        qasm_version (Literal["qasm2", "qasm3"], optional):
            The export version of OpenQASM. Defaults to 'qasm3'.

    Raises:
        ValueError: If the OpenQASM version is invalid.

    Returns:
        QuantumCircuit: The loaded circuit.
    """
    if qasm_version is None:
        qasm_version = qasm_version_detect(qasm_str)

    if qasm_version == "qasm2":
        try:
            return loads_qasm2(qasm_str)
        except QASM2Error as err:
            print(f"| Skip loads from OpenQASM2, due to QASM2Error: {err}")
            return None
    elif qasm_version == "qasm3":
        try:
            return loads_qasm3(qasm_str)
        except QASM3Error as err:
            print(f"| Skip loads from OpenQASM3, due to QASM3Error: {err}")
            return None
    else:
        raise ValueError(f"Invalid qasm version: {qasm_version}")
