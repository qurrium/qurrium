"""Backend Utils (:mod:`qurry.tools.backend.utils`)

For `qiskit-aer` has been seperated from qiskit,
So it needs to be imported differently by trying to import `qiskit-aer` first.

And `qiskit-ibmq-provider` has been deprecated,
but for some user may still need to use it,
so it needs to be imported also differently by trying to import `qiskit-ibm-provider` first.

So this file is used to unify the import point of AerProvider, `IBMProvider`/`IBMQProvider`.
Avoiding the import error occurs on different parts of Qurrium.

"""

from collections.abc import Callable

from qiskit.providers import BackendV2, Backend


def backend_name_getter(back: BackendV2 | Backend | str) -> str:
    """Get the name of backend.

    Args:
        back (BackendV2 | Backend | str): The backend instance.
    Returns:
        str: The name of backend.
    """

    if isinstance(back, str):
        return back
    if isinstance(back, BackendV2):
        return back.name
    if isinstance(back, Callable):
        return back.name()  # type: ignore
    if isinstance(back, Backend):
        return str(back)
    return "unknown_backend"


def shorten_name(name: str, drop: list[str] | None = None, exclude: list[str] | None = None) -> str:
    """Shorten the name of backend.

    Args:
        name (str): The name of backend.
        drop (list[str] | None, optional): The strings to drop from the name. Defaults to [].
        exclude (list[str] | None, optional): The strings to exclude from the name. Defaults to [].

    Returns:
        str: The shortened name of backend.
    """
    if drop is None:
        drop = []
    if exclude is None:
        exclude = []

    if name in exclude:
        return name

    drop = sorted(drop, key=len, reverse=True)
    for _s in drop:
        if _s in name:
            return name.replace(_s, "")

    return name
