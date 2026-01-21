"""EntropyMeasureRandomized - Tales (:mod:`qurry.qurries.entropy_randomized.tales`)"""

from typing import Any, overload, Literal
import numpy as np

from ...qurrium import Tales
from ...capsule import jsonablize


class RandomizedMeasureTales(Tales):
    """The tales for :class:`~qurry.qurries.entropy_randomized.experiment.EMRExperiment` and
    :class:`~qurry.qurries.echo_randomized.experiment.ELRExperiment`.
    """

    @classmethod
    def remain_keys(cls) -> tuple[str, ...]:
        return ("unitary_operator", "bloch_vector")

    @overload
    def __getitem__(
        self, key: Literal["unitary_operator"]
    ) -> dict[int, dict[int, list[list[complex]]]]: ...
    @overload
    def __getitem__(
        self, key: Literal["bloch_vector"]
    ) -> dict[int, dict[int, tuple[float, float, float]]]: ...
    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)

    def export(self) -> dict[str, Any]:
        """Export the serializable data.

        Returns:
            dict[str, Any]: The serializable data.
        """
        unitary_operator = self.get("unitary_operator", {})
        bloch_vector = self.get("bloch_vector", {})

        dedicated = {
            "unitary_operator": {
                n_u_i: {n_u_qi: np.array(op, dtype=str).tolist() for n_u_qi, op in ops.items()}
                for n_u_i, ops in unitary_operator.items()
            },
            "bloch_vector": bloch_vector,
        }
        others = jsonablize({k: v for k, v in self.items() if k not in dedicated})

        return {**dedicated, **others}

    @classmethod
    def ingest(cls, raw_dict: dict[str, Any]):
        """Ingest from a serialized dictionary.

        Args:
            raw_dict (dict[str, Any]): The raw serialized dictionary.
        """
        if set(cls.remain_keys()) - raw_dict.keys():
            raise KeyError(
                "The keys 'unitary_operator' and 'bloch_vector' are required "
                "in the raw dictionary to ingest EMRTales."
            )
        dedicated = {
            "unitary_operator": {
                n_u_i: {
                    n_u_qi: np.array(op_str, dtype=complex).tolist()
                    for n_u_qi, op_str in ops.items()
                }
                for n_u_i, ops in raw_dict["unitary_operator"].items()
            },
            "bloch_vector": {
                n_u_i: {n_u_qi: tuple(vec_list) for n_u_qi, vec_list in vecs.items()}
                for n_u_i, vecs in raw_dict["bloch_vector"].items()
            },
        }
        others = {k: v for k, v in raw_dict.items() if k not in dedicated}

        return cls({**dedicated, **others})
