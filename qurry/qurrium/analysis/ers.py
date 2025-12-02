"""The Entries and Result definitions for analysis. (:mod:`qurry.qurrium.analysis.ers`)"""

from typing import Any, TypeVar
from dataclasses import dataclass, fields
import warnings

from ..json_io import DataExportableLoadable
from ...exceptions import QurryInvalidInherition


@dataclass(frozen=True)
class AnalysisERABC(DataExportableLoadable):
    """Construct the analyze entries's and results's parameters for specific options,
    which should be overwritable by the inherition class of this base class."""

    __name__ = "AnalysisERABC"

    @property
    def fields(self) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(self.__dict__.keys())

    @classmethod
    def dataclass_fields(cls) -> tuple[str, ...]:
        """The fields of arguments."""
        return tuple(f.name for f in fields(cls))

    def asdict(self):
        """The arguments as dictionary."""
        return self.__dict__

    def __post_init__(self):
        """Post-initialization to ensure all fields are present."""
        if self.__name__ == "AnalysisERABC":
            warnings.warn(
                "AnalysisERABC is a base class and should be inherited. "
                "Direct instantiation is discouraged. "
                "If you finish the inherition but still see this warning, "
                "then change the __name__ attribute of your derived class.",
                UserWarning,
            )

        for field_name, field_type in self.__annotations__.items():
            value = getattr(self, field_name)
            if not isinstance(value, field_type):
                raise TypeError(
                    f"Field '{field_name}' expected type '{field_type}', "
                    f"but received '{type(value)}'."
                )

    def pre_export(self) -> dict[str, Any]:
        """Pre-process the results before exporting
        to transform some fields to json-serializable formats.

        Returns:
            A dictionary containing all fields of the results.
        """
        return {field: getattr(self, field) for field in self.fields}

    def export(self) -> dict[str, Any]:
        """Export the results for file writing.

        Returns:
            A tuple containing the class name and the results as a dictionary.
        """
        serialized_data = {"__class__": self.__class__.__name__}
        serialized_data.update(self.pre_export())

        return serialized_data

    @classmethod
    def pre_load(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Pre-process the data before loading
        to recover their type from json-serializable formats.

        Args:
            data (dict[str, Any]): The data to pre-process.

        Returns:
            dict[str, Any]: The pre-processed data.
        """
        return data

    @classmethod
    def load(cls, raw_dict: dict[str, Any]):
        """Load the results from a dictionary.

        Args:
            raw_dict (dict[str, Any]): The data to load.

        Returns:
            The loaded results object.
        """
        classname = raw_dict.pop("__class__", None)
        if classname is None:
            raise ValueError("Data does not contain '__class__' key.")
        if classname != cls.__name__:
            raise ValueError(
                f"Data class '{classname}' does not match expected class '{cls.__name__}'."
            )

        if set(raw_dict.keys()) != set(cls.dataclass_fields()):
            raise ValueError(
                f"Data fields mismatch: expected {cls.dataclass_fields()}, got {set(raw_dict.keys())}."
            )

        return cls(**cls.pre_load(raw_dict))


@dataclass(frozen=True)
class AnalysisMiddlewarePrototype(AnalysisERABC):
    """The middleware information between :func:`~qurry.qurrium.qurrium.QurriumPrototype.analyze`
    and actual post-processing function."""


_RM = TypeVar("_RM", bound=AnalysisMiddlewarePrototype)
"""Type variable for :class:`AnalysisMiddlewarePrototype`."""


@dataclass(frozen=True)
class ProcessEntriesPrototype(AnalysisERABC):
    """The entries for post-processing."""


_PE = TypeVar("_PE", bound=ProcessEntriesPrototype)
"""Type variable for :class:`ProcessEntriesPrototype`."""


def implementation_check_entries(
    middleware_entries: AnalysisMiddlewarePrototype,
    postprocess_entries: ProcessEntriesPrototype,
    analysis_name: str,
) -> None:
    """Check whether the derived class implements all required fields
    from the base class.

    Args:
        middleware_entries (AnalysisMiddlewarePrototype): The middleware entries to check.
        postprocess_entries (ProcessEntriesPrototype): The postprocess entries to check.
        analysis_name (str): The name of the analysis.

    Raises:
        QurryInvalidInherition: If the derived class does not implement all required fields.
    """
    duplicate_fields = (
        set(middleware_entries.fields)
        & set(postprocess_entries.fields)
        & {"serial", "datetime", "log"}
    )
    if len(duplicate_fields) > 0:
        raise QurryInvalidInherition(
            f"{postprocess_entries}, and {middleware_entries} "
            f"should not have same fields: {duplicate_fields} "
            f"for {analysis_name}."
        )


@dataclass(frozen=True, repr=False)
class AnalysisResultsPrototype(AnalysisERABC):
    """The content of the analysis results."""

    def side_product_fields(self) -> tuple[str, ...]:
        """The fields that will be stored as side product."""
        return ()

    def main_and_side_product(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Export the results as two dictionaries: main results and side products.

        Returns:
            A tuple containing the main results and side products as dictionaries.
        """
        main_result = {}
        side_product = {}
        for field in self.fields:
            if field in self.side_product_fields():
                side_product[field] = getattr(self, field)
            else:
                main_result[field] = getattr(self, field)
        return main_result, side_product

    def __repr__(self) -> str:
        """String representation of the AnalyzeResultsPrototype."""
        field_strs = [
            f"{field}={getattr(self, field)!r}"
            for field in self.fields
            if field not in self.side_product_fields()
        ]
        field_strs += [f"side_product_fields={self.side_product_fields()!r}"]
        return f"{self.__class__.__name__}({', '.join(field_strs)})"


_RR = TypeVar("_RR", bound=AnalysisResultsPrototype)
"""Type variable for :class:`AnalyzeResultsPrototype` during transformation."""


def implementation_check_results(
    results: dict[str, _RR],
    available_results_types: dict[str, type[_RR]],
    analysis_name: str,
) -> None:
    """Check whether the derived class implements all required fields
    from the base class.

    Args:
        results (dict[str, AnalyzeResultsPrototype]):
            The results to check.
        available_results_types (dict[str, type[AnalyzeResultsPrototype]]):
            The available results types to check.
        analysis_name (str): The name of the analysis.

    Raises:
        QurryInvalidInherition: If the derived class does not implement all required fields.
    """
    if any(
        rt_name not in available_results_types
        or not isinstance(results[rt_name], available_results_types[rt_name])
        for rt_name in results
    ):
        raise TypeError(
            f"Results types mismatch for {analysis_name}: "
            f"expected {list(available_results_types.keys())}, "
            f"got {list(results.keys())}."
        )
