"""Import Fake (:mod:`qurry.tools.backend.import_fake`)

This file is used to unify the import point of FakeProvider, FakeBackend/FakeBackendV2
from qiskit.providers.fake_provider and qiskit_ibm_runtime.fake_provider.
Avoiding the import error occurs on different parts of Qurry.

"""

from typing import Literal, Type, Union, Optional, overload
import warnings

from qiskit.providers import BackendV2, Backend

from .utils import backend_name_getter, shorten_name
from ...exceptions import QurryDependenciesNotWorking, QurryDependenciesFailureError

# pylint: disable=ungrouped-imports
ImportPointType = Literal[
    "qiskit_ibm_runtime.fake_provider",
    "qiskit.providers.fake_provider",
]
ImportPointOrder: list[ImportPointType] = [
    "qiskit_ibm_runtime.fake_provider",
    "qiskit.providers.fake_provider",
]
FAKE_BACKENDV2_SOURCES: dict[ImportPointType, Optional[Type[BackendV2]]] = {}
FAKE_PROVIDERFORV2_SOURCES: dict[
    ImportPointType,
    Optional[Union[Type["FakeProviderForBackendV2Dep"], Type["FakeProviderForBackendV2Indep"]]],
] = {}
FAKE_VERSION_INFOS: dict[ImportPointType, Optional[str]] = {}
FAKE_IMPORT_ERROR_INFOS: dict[ImportPointType, ImportError] = {}

QISKIT_IBM_RUNTIME_ISSUE_1318 = (
    "The version of 'qiskit-ibm-runtime' is 0.18.0, "
    "FackBackend is not working in this version for this issue: "
    "https://github.com/Qiskit/qiskit-ibm-runtime/issues/1318."
    "You need to change the version of 'qiskit-ibm-runtime' to access FakeBackend"
)

try:
    from qiskit_ibm_runtime import __version__ as qiskit_ibm_runtime_version  # type: ignore

    major, minor, _ = qiskit_ibm_runtime_version.split(".")
    IS_V1_REMOVE = int(major) >= 0 and int(minor) >= 31
    if major == "18" and minor == "0":
        warnings.warn(
            QISKIT_IBM_RUNTIME_ISSUE_1318,
            category=QurryDependenciesNotWorking,
        )

    from qiskit_ibm_runtime.fake_provider import (  # type: ignore
        FakeProviderForBackendV2 as FakeProviderForBackendV2Indep,  # type: ignore
    )
    from qiskit_ibm_runtime.fake_provider.fake_backend import (  # type: ignore
        FakeBackendV2 as FakeBackendV2Indep,  # type: ignore
    )

    FAKE_BACKENDV2_SOURCES["qiskit_ibm_runtime.fake_provider"] = FakeBackendV2Indep
    FAKE_PROVIDERFORV2_SOURCES["qiskit_ibm_runtime.fake_provider"] = FakeProviderForBackendV2Indep
    FAKE_VERSION_INFOS["qiskit_ibm_runtime.fake_provider"] = qiskit_ibm_runtime_version

except ImportError as err:
    FAKE_IMPORT_ERROR_INFOS["qiskit_ibm_runtime.fake_provider"] = err

if len(FAKE_BACKENDV2_SOURCES) == 0:
    try:
        from qiskit.providers.fake_provider import (
            FakeProviderForBackendV2 as FakeProviderForBackendV2Dep,  # type: ignore
            FakeBackendV2 as FakeBackendV2Dep,  # type: ignore
        )
        from qiskit import __version__ as qiskit_version

        FAKE_BACKENDV2_SOURCES["qiskit.providers.fake_provider"] = FakeBackendV2Dep
        FAKE_PROVIDERFORV2_SOURCES["qiskit.providers.fake_provider"] = FakeProviderForBackendV2Dep
        FAKE_VERSION_INFOS["qiskit.providers.fake_provider"] = qiskit_version

    except ImportError as err:
        FAKE_IMPORT_ERROR_INFOS["qiskit.providers.fake_provider"] = err


def get_default_fake_provider() -> Optional[ImportPointType]:
    """Get the default fake provider.

    Returns:
        ImportPointType: The default fake provider.
    """
    for source in ImportPointOrder:
        if source in FAKE_PROVIDERFORV2_SOURCES:
            return source
    return None


FAKE_DEFAULT_SOURCE: Optional[ImportPointType] = get_default_fake_provider()


LUCKY_MSG = """
No fake provider available. It may be caused by version conflict. 
For qiskit<1.0.0 please install qiskit-ibm-runtime<0.21.0 by 
'pip install qiskit-ibm-runtime<0.21.0'. 
If you are still using qiskit 0.46.X and lower originally, 
then install newer qiskit-ibm-runtime at same time, 
please check whether the version of qiskit 
has been updated to 1.0 by the installation 
because since qiskit-ibm-runtime 0.21.0+ 
has been updated its dependency to qiskit 1.0. 
If you already have qiskit-ibm-runtimes installed and lower than 0.21.0, 
it is only available to use qiskit 0.46.X as dependency 
for the migration of fake_provider is not completed around this version. 
Many of the fake backends are not available in qiskit-ibm-runtime. 
(This made me a lot problem to handle the fake backends in Qurry.) 
(If you see this error raised, good luck to you to fix environment. :smile:.) 
""".replace(
    "\n", " "
).strip()


@overload
def fack_backend_loader() -> tuple[dict[str, str], dict[str, Backend]]: ...


@overload
def fack_backend_loader() -> tuple[dict[str, str], dict[str, Backend]]: ...


def fack_backend_loader():
    """Load the fake backend.

    Args:
        version (str, optional): The version of fake backend. Defaults to None.
        "v1" for FakeProvider, "v2" for FakeProviderForBackendV2.

    Returns:
        tuple[dict[str, str], dict[str, Backend]]:
            The callsign of fake backend,
            the fake backend dict,
            the fake provider.
    """

    if FAKE_DEFAULT_SOURCE is None:
        warnings.warn(LUCKY_MSG, category=QurryDependenciesNotWorking)
        return {}, {}

    _fake_provider_v2_becalled = FAKE_PROVIDERFORV2_SOURCES.get(FAKE_DEFAULT_SOURCE, None)

    if _fake_provider_v2_becalled is None:
        raise QurryDependenciesFailureError(LUCKY_MSG)
    try:
        _fake_provider = _fake_provider_v2_becalled()
    except FileNotFoundError as err1318:
        raise QurryDependenciesFailureError(QISKIT_IBM_RUNTIME_ISSUE_1318) from err1318

    backend_fake: dict[str, Backend] = {
        backend_name_getter(b): b for b in _fake_provider.backends()
    }
    backend_fake_callsign = {shorten_name(bn, ["_v2"]): bn for bn in backend_fake}
    backend_fake_callsign["fake_qasm"] = "fake_qasm_simulator"
    return backend_fake_callsign, backend_fake
