"""Backend Wrapper (:mod:`qurry.tools.backend.backend_manager`)"""

from typing import Union, Literal, Callable, Optional
from random import random
import warnings

from qiskit.providers import Backend

from .import_simulator import SIM_DEFAULT_SOURCE as sim_default_source, GeneralSimulator
from .import_fake import FAKE_BACKENDV2_SOURCES as fake_default_source, fack_backend_loader

from ...capsule.hoshi import Hoshi
from ...exceptions import QurryDeprecatedWarning


BackendDict = dict[
    Union[Literal["real", "sim", "fake", "extra"], str],
    dict[str, Backend],
]
"""The dictionary of backends."""


BackendCallSignDict = dict[
    Union[Literal["real", "sim", "fake", "extra"], str],
    dict[str, str],
]
"""The dictionary of backend callsign."""


def _statesheet_preparings(
    check_msg: Hoshi,
    desc: str,
    backs: list[str],
    backs_callsign: dict[str, str],
    is_aer_gpu: bool,
):
    backs_len = len(backs)
    check_msg.divider()
    check_msg.h4(desc)
    if "Simulator" in desc:
        check_msg.newline(
            {
                "type": "itemize",
                "description": "Aer GPU",
                "value": is_aer_gpu,
                "ljust_description_filler": ".",
            }
        )
        check_msg.newline(
            {
                "type": "itemize",
                "description": "Simulator Provider by",
                "value": sim_default_source,
                "ljust_description_filler": ".",
            }
        )
    if backs_len == 0:
        check_msg.newline(
            {
                "type": "txt",
                "listing_level": 2,
                "text": (
                    "No Backends Available."
                    + (
                        " Choose fake version when initializing the backend wrapper."
                        if "Fake" in desc
                        else ""
                    )
                    + (
                        (
                            " Real backends need to be loaded by 'BackendManager' "
                            + "instead of 'BackendWrapper'."
                        )
                        if "IBM" in desc
                        else ""
                    )
                ),
            }
        )
    else:
        for i in range(0, backs_len, 3):
            tmp_backs = backs[i : i + 3]
            tmp_backs_str = ", ".join(tmp_backs) + ("," if len(tmp_backs) == 3 else "")
            check_msg.newline(
                {
                    "type": "txt",
                    "listing_level": 2,
                    "text": tmp_backs_str,
                }
            )

    if len(backs_callsign) == 0:
        check_msg.newline(
            {
                "type": "txt",
                "listing_level": 2,
                "text": "No Callsign Added",
            }
        )
    else:
        check_msg.newline(
            {
                "type": "itemize",
                "description": f"Available {desc} Backends Callsign",
            }
        )
        for k, v in backs_callsign.items():
            check_msg.newline(
                {
                    "type": "itemize",
                    "description": f"{k}",
                    "value": f"{v}",
                    "listing_level": 2,
                    "ljust_description_filler": ".",
                }
            )


class BackendWrapper:
    """A wrapper for :class:`qiskit.providers.Backend` to provide more convenient way to use."""

    @staticmethod
    def _hint_ibmq_sim(name: str) -> str:
        return "ibm" + name if "ibm" not in name else name

    def __init__(
        self,
    ) -> None:

        self.is_aer_gpu = False
        backend_fake_callsign, backend_fake = fack_backend_loader()

        self.backend_dict: BackendDict = {
            "sim": {"sim": GeneralSimulator()},
            "real": {},
            "fake": {**backend_fake},
            "extra": {},
        }
        self.backend_callsign_dict: BackendCallSignDict = {
            "sim": {},
            "real": {},
            "fake": {**backend_fake_callsign},
            "extra": {},
        }

        if sim_default_source == "qiskit.providers.basicaer":
            warnings.warn(
                "The qiskit.providers.basicaer is an outdated module, "
                + "you should migrate to qiskit-aer, "
                + "or update qiskit to the latest version.",
                category=QurryDeprecatedWarning,
            )

        if hasattr(self.backend_dict["sim"]["sim"], "available_devices"):
            assert isinstance(
                self.backend_dict["sim"]["sim"].available_devices, Callable  # type: ignore
            ), "The available_devices should be a callable."

            self.is_aer_gpu = (
                "GPU" in self.backend_dict["sim"]["sim"].available_devices()  # type: ignore
            )
            self.backend_dict["sim"]["sim"].set_options(device="GPU")  # type: ignore
            assert self.backend_dict["sim"]["sim"].options.device == "GPU", (  # type: ignore
                "GPU is not available, consider to check your CUDA installation."
            )

    def __repr__(self):
        repr_str = f"<{self.__class__.__name__}("
        repr_str += f'sim="{sim_default_source}", '
        repr_str += f'fake="{fake_default_source}"'
        repr_str += ")>"
        return repr_str

    def make_callsign(
        self,
        sign: str = "Galm 2",
        who: str = "solo_wing_pixy",
    ) -> None:
        """Make a callsign for backend.

        Args:
            sign (str, optional): The callsign.
            who (str, optional): The backend.

        Raises:
            ValueError: If the callsign already exists.
            ValueError: If the backend is unknown.
        """

        if sign == "Galm 2" or who == "solo_wing_pixy":
            if random() <= 0.2:
                print(
                    "Those who survive a long time on the battlefield "
                    + "start to think they're invincible. I bet you do, too, Buddy."
                )
        for avaiable_type in ["real", "sim", "fake", "extra"]:
            if sign in self.backend_callsign_dict[avaiable_type]:
                raise ValueError(f"'{sign}' callsign already exists.")
        for avaiable_type in ["real", "sim", "fake", "extra"]:
            if who in self.backend_dict[avaiable_type]:
                self.backend_callsign_dict[avaiable_type][sign] = who
                return
        raise ValueError(f"'{who}' unknown backend.")

    @property
    def available_backends(self) -> BackendDict:
        """The available backends."""
        return self.backend_dict

    @property
    def available_backends_callsign(self) -> BackendCallSignDict:
        """The available backends callsign."""
        return self.backend_callsign_dict

    @property
    def available_aer(self) -> list[str]:
        """The available aer backends."""
        return list(self.backend_dict["sim"].keys())

    @property
    def available_aer_callsign(self) -> list[str]:
        """The available aer backends callsign."""
        return list(self.backend_callsign_dict["sim"].keys())

    @property
    def available_ibmq(self) -> list[str]:
        """The available ibmq/ibm backends."""
        return list(self.backend_dict["real"].keys())

    @property
    def available_ibmq_callsign(self) -> list[str]:
        """The available ibmq/ibm backends callsign."""
        return list(self.backend_callsign_dict["real"].keys())

    @property
    def available_fake(self) -> list[str]:
        """The available fake backends."""
        return list(self.backend_dict["fake"].keys())

    @property
    def available_fake_callsign(self) -> list[str]:
        """The available fake backends callsign."""
        return list(self.backend_callsign_dict["fake"].keys())

    def statesheet(self):
        """The statesheet of backend wrapper."""
        check_msg = Hoshi(
            [
                ("divider", 60),
                ("h3", "BackendWrapper Statesheet"),
            ],
            ljust_description_len=35,
            ljust_description_filler=".",
        )

        for desc, backs, backs_callsign in [
            ("Simulator", self.available_aer, self.backend_callsign_dict["sim"]),
            ("IBM", self.available_ibmq, self.backend_callsign_dict["real"]),
            ("Fake", self.available_fake, self.backend_callsign_dict["fake"]),
            (
                "Extra",
                self.available_backends["extra"],
                self.backend_callsign_dict["extra"],
            ),
        ]:
            _statesheet_preparings(
                check_msg,
                desc,
                backs,
                backs_callsign,
                self.is_aer_gpu,
            )

        return check_msg

    def add_backend(
        self,
        name: str,
        backend: Backend,
        callsign: Optional[str] = None,
    ) -> None:
        """Add a backend to backend wrapper.

        Args:
            name (str): The name of backend.
            backend (Backend): The backend.
            callsign (Optional[str], optional): The callsign of backend. Defaults to None.
        """

        if not isinstance(backend, Backend):
            raise TypeError("The backend should be a instance of 'qiskit.providers.Backend'")

        if name in self.backend_dict["extra"]:
            raise ValueError(f"'{name}' backend already exists.")

        self.backend_dict["extra"][name] = backend
        if callsign is not None:
            self.backend_callsign_dict["extra"][callsign] = name

    def __call__(
        self,
        backend_name: str,
    ) -> Backend:
        for avaiable_type in ["real", "sim", "fake", "extra"]:
            if backend_name in self.backend_dict[avaiable_type]:
                return self.backend_dict[avaiable_type][backend_name]
            if backend_name in self.backend_callsign_dict[avaiable_type]:
                return self.backend_dict[avaiable_type][
                    self.backend_callsign_dict[avaiable_type][backend_name]
                ]

        raise ValueError(f"'{backend_name}' unknown backend or backend callsign.")
