"""Analysis Instance (:mod:`qurry.qurrium.analysis`)"""

from typing import Optional, NamedTuple, Iterable, Any, Generic, TypeVar, Type
from abc import abstractmethod
from pathlib import Path
import json


from ...capsule import jsonablize, DEFAULT_ENCODING
from ...capsule.hoshi import Hoshi
from ...exceptions import QurryInvalidInherition
from ...tools.datetime import current_time


_RI = TypeVar("_RI", bound=NamedTuple)
"""The input type of the analysis."""
_RC = TypeVar("_RC", bound=NamedTuple)
"""The content type of the analysis."""


class AnalysisPrototype(Generic[_RI, _RC]):
    """The instance for the analysis of :cls:`QurryExperiment`."""

    __name__ = "AnalysisPrototype"

    serial: int
    """Serial Number of analysis."""
    datetime: str
    """Written time of analysis."""
    log: dict[str, Any]
    """Other info will be recorded."""

    @classmethod
    @abstractmethod
    def input_type(cls) -> Type[_RI]:
        """The input type of the analysis."""
        raise NotImplementedError("input_type must be implemented in subclass.")

    @property
    def input_instance(self) -> Type[_RI]:
        """The input instance of the analysis."""
        return self.input_type()

    @classmethod
    @abstractmethod
    def content_type(cls) -> Type[_RC]:
        """The content type of the analysis."""
        raise NotImplementedError("content_type must be implemented in subclass.")

    @property
    def content_instance(self) -> Type[_RC]:
        """The content instance of the analysis."""
        return self.content_type()

    def __eq__(self, other) -> bool:
        """Check if two analysis instances are equal."""
        if isinstance(other, self.__class__):
            return self.input == other.input
        return False

    @property
    @abstractmethod
    def side_product_fields(self) -> Iterable[str]:
        """The fields that will be stored as side product."""
        raise NotImplementedError("side_product_fields must be implemented in subclass.")

    def __init__(
        self,
        *,
        serial: int,
        log: Optional[dict[str, Any]] = None,
        datatime: Optional[str] = None,
        **other_kwargs,
    ):
        duplicate_fields = (
            set(self.input_instance._fields)
            & set(self.content_instance._fields)
            & {"serial", "datetime", "log"}
        )
        if len(duplicate_fields) > 0:
            raise QurryInvalidInherition(
                f"{self.input_instance} and {self.content_instance} "
                f"should not have same fields: {duplicate_fields} "
                f"for {self.__name__}."
            )

        self.serial = serial
        self.datetime = current_time() if datatime is None else datatime
        self.log = log if isinstance(log, dict) else {}

        lost_fields = [
            k
            for k in self.input_instance._fields + self.content_instance._fields
            if k not in other_kwargs
        ]
        if len(lost_fields) > 0:
            raise QurryInvalidInherition(
                f"{self.__name__} should have all fields in "
                f"{self.input_instance.__name__} and {self.content_instance.__name__}, "
                f"but lost fields: {lost_fields}."
            )
        self.input: _RI = self.input_instance._make(
            other_kwargs.pop(k) for k in self.input_instance._fields
        )
        """The input of the analysis."""
        self.content: _RC = self.content_instance._make(
            other_kwargs.pop(k) for k in self.content_instance._fields
        )
        """The content of the analysis."""
        self.outfields = other_kwargs

    def __repr__(self) -> str:
        return (
            f"<{self.__name__}("
            + f"serial={self.serial}, {self.input}, {self.content}), "
            + f"unused_args_num={len(self.outfields)}>"
        )

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(
                f"<{self.__name__}("
                + f"serial={self.serial}, {self.input}, {self.content}), "
                + f"unused_args_num={len(self.outfields)}>"
            )
        else:
            with p.group(2, f"<{self.__name__}(", ")>"):
                p.breakable()
                p.text(f"serial={self.serial},")
                p.breakable()
                p.text(f"{self.input},")
                p.breakable()
                p.text(f"{self.content}),")
                p.breakable()
                p.text(f"unused_args_num={len(self.outfields)}")
                p.breakable()

    def statesheet(self, hoshi: bool = False) -> Hoshi:
        """Generate the state sheet of the analysis.

        Args:
            hoshi (bool, optional):
                If True, show Hoshi name in statesheet. Defaults to False.
        Returns:
            Hoshi: The state sheet of the analysis.
        """
        info = Hoshi(
            [
                ("h1", f"{self.__name__} with serial={self.serial}"),
            ],
            name="Hoshi" if hoshi else "QurryAnalysisSheet",
        )
        info.newline(("itemize", "serial", self.serial, "", 1))
        info.newline(("itemize", "datetime", self.datetime, "", 1))

        info.newline(("itemize", "input"))
        for k, v in self.input._asdict().items():
            info.newline(("itemize", str(k), str(v), (), 2))

        info.newline(
            ("itemize", "outfields", len(self.outfields), "Number of unused arguments.", 1)
        )
        for k, v in self.outfields.items():
            info.newline(("itemize", str(k), str(v), "", 2))

        info.newline(("itemize", "content"))
        for k, v in self.content._asdict().items():
            info.newline(("itemize", str(k), str(v), "", 2))

        info.newline(("itemize", "log"))
        for k, v in self.log.items():
            info.newline(("itemize", str(k), str(v), "", 2))

        return info

    def export(self, jsonable: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        """Export the analysis as main and side product dict.

        Args:
            jsonable (bool, optional):
                If True, export as jsonable dict. Defaults to True.
                If False, export as normal dict.

        .. code-block:: python
            main = { ...quantities, 'input': { ... }, 'header': { ... }, }
            side = { 'dummyz1': ..., 'dummyz2': ..., ..., 'dummyzm': ... }

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: `main` and `side` product dict.

        """

        tales = {}
        main = {}
        for k, v in self.content._asdict().items():
            if k in self.side_product_fields:
                tales[k] = v
            else:
                main[k] = v
        main["input"] = self.input._asdict()
        main["header"] = {
            "serial": self.serial,
            "datetime": self.datetime,
            "log": self.log,
        }

        if jsonable:
            return jsonablize(main), jsonablize(tales)
        return main, tales

    @classmethod
    def deprecated_fields_converts(
        cls, main: dict[str, Any], side: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert deprecated fields to new fields.

        This method should be implemented in the subclass if there are deprecated fields
        that need to be converted.

        Args:
            main (dict[str, Any]): The main product dict.
            side (dict[str, Any]): The side product dict.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:
                The converted main and side product dicts.
        """
        return main, side

    @classmethod
    def load(cls, main: dict[str, Any], side: dict[str, Any]):
        """Read the analysis from main and side product dict.

        Args:
            main (dict[str, Any]): The main product dict.
            side (dict[str, Any]): The side product dict.

        Returns:
            AnalysisPrototype: The analysis instance.
        """
        main, side = cls.deprecated_fields_converts(main, side)
        content = {k: v for k, v in main.items() if k not in ("input", "header")}
        serial = main["header"].get("serial", 0)
        log = main["header"].get("log", {})
        datetime = main["header"].get("datetime", current_time())
        instance = cls(
            serial=serial, log=log, datatime=datetime, **main["input"], **content, **side
        )
        return instance

    @classmethod
    def read(cls, file_index: dict[str, str], save_location: Path):
        """Read the analysis from file index.

        Args:
            file_index (dict[str, str]): The file index.
            save_location (Path): The save location.

        Returns:
            dict[str, AnalysisPrototype]: The analysis instances in dictionary.
        """

        export_material_set: dict[str, dict[str, dict[str, Any]]] = {
            "reports": {},
            "tales_report": {},
        }
        for filekey, filename in file_index.items():
            filekey_split = filekey.split(".")
            if filekey == "reports":
                with open(save_location / filename, "r", encoding=DEFAULT_ENCODING) as f:
                    tmp = json.load(f)
                    export_material_set["reports"] = tmp["reports"]

            elif filekey_split[0] == "reports" and filekey_split[1] == "tales":
                with open(save_location / filename, "r", encoding=DEFAULT_ENCODING) as f:
                    export_material_set["tales_report"][filekey_split[2]] = json.load(f)

        mains = export_material_set["reports"]
        sides = {rk: {} for rk in export_material_set["reports"]}

        for tk, tv in export_material_set["tales_report"].items():
            for rk, rv in tv.items():
                if rk not in sides:
                    sides[rk] = {}
                sides[rk][tk] = rv

        return {int(k) if k.isdigit() else k: cls.load(v, sides[k]) for k, v in mains.items()}
