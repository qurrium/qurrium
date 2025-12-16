"""QuantityContainer (:mod:`qurry.qurrium.multimanager.multiquantity`)

A container for result of
:meth:`~qurry.qurrium.multimanager.multimanager.MultiManager.analyze` for
:class:`~qurry.qurrium.multimanager.multimanager.MultiManager`.
"""

from typing import Any
from pathlib import Path
import json

from ..utils import ExportFolderNaming
from ..utils.iocontrol import RJUST_LEN, serial_naming
from ...capsule import (
    CustomDict,
    jsonablize,
    key_tuple_loads,
    DEFAULT_ENCODING,
    DEFAULT_INDENT,
    DEFAULT_MODE,
    quick_json_write,
)


def multimanager_report_naming(
    quantities_container: "MutltiQuantityInfo",
    analysis_name: str,
    no_serialize: bool,
) -> str:
    """Naming the report in the quantity container.

    Args:
        analysis_name (str):
            The name of the analysis.
        no_serialize (bool):
            Whether to serialize the analysis.
        quantities_container (QuantityContainer):
            The container of the quantities.

    Returns:
        str: The name of the quantity container.
    """
    all_existing = quantities_container.keys()
    if no_serialize:
        if analysis_name in all_existing:
            raise ValueError(
                f"The analysis name '{analysis_name}' already exists in the quantities container. "
                "Please choose a different name or remove the existing report."
            )
        return f"{analysis_name}"

    repeat_times = 0

    proposal_name = serial_naming(analysis_name, repeat_times, RJUST_LEN)
    while proposal_name in all_existing:
        repeat_times += 1
        proposal_name = serial_naming(analysis_name, repeat_times, RJUST_LEN)

    return proposal_name


class MutltiQuantityInfo(CustomDict[str, dict[tuple[str, ...], list[tuple[str, int]]]]):
    """The container for quantities of analysis for
    :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`."""

    __name__ = "MutltiQuantityInfo"

    def report_naming(self, analysis_name: str, no_serialize: bool) -> str:
        """Naming the report in the quantity container.

        Args:
            analysis_name (str):
                The name of the analysis.
            no_serialize (bool):
                Whether to serialize the analysis.

        Returns:
            str: The name of the quantity container.
        """
        return multimanager_report_naming(
            quantities_container=self,
            analysis_name=analysis_name,
            no_serialize=no_serialize,
        )

    def content_dumping(self) -> dict[str, Any]:
        """Get the content to be written to files.

        Returns:
            dict[str, Any]: The content to be written to files.
        """
        return jsonablize(self)

    def write(self, save_location: Path, summoner_name: str) -> dict[str, str]:
        """Write the beforewards data to files.

        Args:
            save_location (Path): The location of MultiManager.
            summoner_name (str): The name of MultiManager.

        Returns:
            dict[str, str]: The index of saved files.
        """
        exported_content = self.content_dumping()

        quick_json_write(
            exported_content,
            "multiquantity.json",
            DEFAULT_MODE,
            indent=DEFAULT_INDENT,
            encoding=DEFAULT_ENCODING,
            save_location=save_location,
            mute=True,
        )

        return {"multiquantity": str(Path(summoner_name) / "multiquantity.json")}

    @classmethod
    def content_loading(cls, raw_dict: dict[str, Any]) -> dict[str, Any]:
        """Process the serialized content from the method :meth:`content_writing`
        Handle the raw read dictionary with specific structure,
        which is same with the one used in :meth:`content_dumping`.

        Args:
            raw_dict (dict[str, Any]): The raw dictionary.

        Returns:
            dict[str, Any]: The loaded content.
        """

        return {
            key: {key_tuple_loads(k): v for k, v in value.items()}
            for key, value in raw_dict.items()
        }

    @classmethod
    def read(cls, file_index: dict[str, str], naming_complex: ExportFolderNaming):
        """Read the exported experiment file.

        Args:
            file_index (dict[str, str]): The index of exported experiment file.
            naming_complex (ExportFolderNaming): The naming complex of MultiManager.
        """

        if "multiquantity" not in file_index:
            raise KeyError("The file index does not contain 'multiquantity' key.")

        with open(
            naming_complex.save_location / file_index["multiquantity"],
            "r",
            encoding=DEFAULT_ENCODING,
        ) as f:
            multiquantity_data = json.load(f)

        return cls(cls.content_loading(multiquantity_data))

    def register(
        self,
        analysis_source_info: list[tuple[tuple[str, ...], str, int]],
        analysis_name: str,
        no_serialize: bool,
    ) -> str:
        """Register the analysis result into the container.

        Args:
            analysis_source_info (list[tuple[tuple[str, ...], str, int]]):
                The analysis source information.
                Each element is a tuple of
                (source_exp_tags, source_exp_id, source_quantity_index).
            analysis_name (str):
                The name of the analysis.
            no_serialize (bool):
                Whether to serialize the analysis.

        Returns:
            str: The name of the quantity container.
        """

        report_name = self.report_naming(analysis_name, no_serialize)
        assert report_name not in self, (
            f"The report name '{report_name}' already exists in the quantity container. "
            "This should not happen, please report a bug."
        )
        self[report_name] = {}

        for source_exp_tags, source_exp_id, source_quantity_index in analysis_source_info:
            if source_exp_tags not in self[report_name]:
                self[report_name][source_exp_tags] = []
            self[report_name][source_exp_tags].append((source_exp_id, source_quantity_index))

        return report_name
