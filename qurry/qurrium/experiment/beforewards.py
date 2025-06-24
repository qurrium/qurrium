"""Experiment - Beforewards (:mod:`qurry.qurrium.experiment.beforewards`)"""

import json
from typing import Optional, NamedTuple, Any, Union
from collections.abc import Hashable
from pathlib import Path

from qiskit import QuantumCircuit

from ..utils.qasm import qasm_loads
from ...capsule import DEFAULT_ENCODING

V5_TO_V7_FIELD = {
    "jobID": "job_id",
    "expName": "exp_name",
    "sideProduct": "side_product",
}
DEPRECATED_PROPERTIES = ["figTranspiled", "fig_original", "exp_name"]


def v5_to_v7_field_transpose(advent: dict[str, Any]) -> dict[str, Any]:
    """Transpose the v5 field to v7 field."""
    for k, nk in V5_TO_V7_FIELD.items():
        if k in advent:
            advent[nk] = advent.pop(k)
    return advent


def v7_to_v11_field_transpose(advent: dict[str, Any]) -> dict[str, Any]:
    """Transpose the v7 field to v11 field."""

    if "job_id" in advent:
        if isinstance(advent["job_id"], str):
            advent["job_id"] = [advent["job_id"]]
        elif isinstance(advent["job_id"], list):
            advent["job_id"] = advent["job_id"]
        else:
            raise TypeError("job_id must be str or list[str].")

    return advent


class Before(NamedTuple):
    """The data of experiment will be independently exported in the folder 'advent',
    which generated before the experiment.
    """

    # Experiment Preparation
    target: list[tuple[Hashable, Union[QuantumCircuit, str]]]
    """The target circuits of experiment."""
    target_qasm: list[tuple[str, str]]
    """The OpenQASM of target circuits."""
    circuit: list[QuantumCircuit]
    """The transpiled circuits of experiment."""
    circuit_qasm: list[str]
    """The OpenQASM of transpiled circuits."""

    # Export data
    job_id: list[str]
    """ID of job for pending on real machine (IBMQBackend)."""

    # side product
    side_product: dict[str, Any]
    """The data of experiment will be independently exported in the folder 'tales'."""

    @staticmethod
    def default_value():
        """These default value are used for autofill the missing value."""
        return {
            "target": [],
            "target_qasm": [],
            "circuit": [],
            "circuit_qasm": [],
            "job_id": [],
            "side_product": {},
        }

    @classmethod
    def read(
        cls,
        file_index: dict[str, str],
        save_location: Path,
    ) -> "Before":
        """Read the exported experiment file.

        Args:
            file_index (dict[str, str]): The index of exported experiment file.
            save_location (Path): The location of exported experiment file.

        Returns:
            tuple[dict[str, Any], "Before", dict[str, Any]]:
                The experiment's arguments,
                the experiment's common parameters,
                and the experiment's side product.
        """
        raw_data = {}
        with open(save_location / file_index["advent"], "r", encoding=DEFAULT_ENCODING) as f:
            raw_data = json.load(f)

        advent: dict[str, Any] = raw_data["adventures"]
        for k in DEPRECATED_PROPERTIES:
            advent.pop(k, None)
        advent = v5_to_v7_field_transpose(advent)
        advent = v7_to_v11_field_transpose(advent)
        for k, dv in cls.default_value().items():
            if k not in advent:
                advent[k] = dv

        assert "side_product" in advent, "The side product is not found."

        for filekey, filename in file_index.items():
            filekeydiv = filekey.split(".")
            if filekeydiv[0] == "tales":
                with open(save_location / filename, "r", encoding=DEFAULT_ENCODING) as f:
                    advent["side_product"][filekeydiv[1]] = json.load(f)

        return cls(**advent)

    def export(
        self,
        export_transpiled_circuit: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Export the experiment's data before executing.

        Args:
            export_circuit (bool, optional):
                Whether to export the transpiled circuit as txt. Defaults to False.
                for It's space-saving purpose and performance improvement.
                When set to True, the transpiled circuit will be draw as txt.
                Otherwise, the circuit will be not exported but circuit qasm remains.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]:
                The experiment's arguments,
                and the experiment's side product.
        """

        adventures = {
            "target": self.target,
            "target_qasm": self.target_qasm,
            "circuit": self.circuit if export_transpiled_circuit else [],
            "circuit_qasm": self.circuit_qasm,
            "job_id": self.job_id,
        }

        return adventures, self.side_product

    def revive_circuit(self, replace_circuits: bool = False) -> list[QuantumCircuit]:
        """Revive the circuit from the qasm, return the revived circuits.

        Args:
            replace_circuits (bool, optional): Whether to replace the circuits. Defaults to False.

        Raises:
            ValueError: If the .circuit is not empty.

        Returns:
            list[QuantumCircuit]: The revived circuits.
        """
        revived_circuits = []
        if len(self.circuit) != 0:
            if replace_circuits:
                self.circuit.clear()
            else:
                raise ValueError(".circuit is not empty.")
        is_none_circuits = []
        for i, qasm in enumerate(self.circuit_qasm):
            tmp_circ = qasm_loads(qasm)
            revived_circuits.append(tmp_circ)
            if tmp_circ is None:
                is_none_circuits.append(i)
        if len(is_none_circuits) != 0:
            print(f"The circuits {is_none_circuits} are not revived.")
        return revived_circuits

    def revive_target(self, replace_target: bool = False) -> dict[Hashable, QuantumCircuit]:
        """Revive the target circuits from the qasm, return the revived target.

        Args:
            replace_target (bool, optional):
                Whether to replace the target circuits. Defaults to False.

        Raises:
            ValueError: If the .target is not empty.

        Returns:
            dict[Hashable, QuantumCircuit]: The revived target circuits.
        """
        revived_target = {}
        if len(self.target) != 0:
            if replace_target:
                self.target.clear()
            else:
                raise ValueError("The target is not empty.")
        for key, qasm in self.target_qasm:
            revived_target[key] = QuantumCircuit.from_qasm_str(qasm)
        return revived_target


def create_beforewards(beforewards: Optional[Before]) -> Before:
    """Create a Beforewards object.

    Args:
        beforewards (Optional[Before]):
            The Beforewards object to create. Defaults to None.
    Returns:
        Before: The Beforewards object.
    Raises:
        TypeError: If 'beforewards' is not a Before object or None.
    """

    if beforewards is None:
        return Before(
            target=[],
            target_qasm=[],
            circuit=[],
            circuit_qasm=[],
            job_id=[],
            side_product={},
        )
    if isinstance(beforewards, Before):
        return beforewards

    raise TypeError(f"beforewards must be a Before object or None, but got {type(beforewards)}.")
