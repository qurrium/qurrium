"""EntropyMeasureRandomized - Tales (:mod:`qurry.qurrent.randomized_measure.tales`)"""

from typing import TypedDict, Any

from ...qurrium import Tales
from ...capsule import jsonablize


class EntropyMeasureTalesTypes(TypedDict):
    """The typed dictionary for :class:`EntropyMeasureTales`."""

    unitary_operator: dict[int, dict[int, list[list[complex]]]]
    """The dictionary of unitary operators."""
    bloch_vector: dict[int, dict[int, tuple[float, float, float]]]
    """The dictionary of bloch vectors."""


class EntropyMeasureTales(Tales[EntropyMeasureTalesTypes]):
    """The tales for :class:`~qurry.qurrent.randomized_measure.experiment.EMRExperiment` and
    :class:`~qurry.qurrech.randomized_measure.experiment.ELRExperiment`."""

    @classmethod
    def remain_keys(cls) -> tuple[str, ...]:
        return ("unitary_operator", "bloch_vector")

    def export(self) -> dict[str, Any]:
        """Export the serializable data.

        Returns:
            dict[str, Any]: The serializable data.
        """
        dedictaed = {
            "unitary_operator": {
                n_u_i: {
                    n_u_qi: str(self["unitary_operator"][n_u_i][n_u_qi]) for n_u_qi in ops.items()
                }
                for n_u_i, ops in self["unitary_operator"].items()
            },
            "bloch_vector": self["bloch_vector"],
        }
        others = jsonablize({k: v for k, v in self.items() if k not in dedictaed})

        return {**dedictaed, **others}

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
                n_u_i: {n_u_qi: complex(op_str) for n_u_qi, op_str in ops.items()}
                for n_u_i, ops in raw_dict["unitary_operator"].items()
            },
            "bloch_vector": {
                n_u_i: {n_u_qi: tuple(vec_list) for n_u_qi, vec_list in vecs.items()}
                for n_u_i, vecs in raw_dict["bloch_vector"].items()
            },
        }
        others = {k: v for k, v in raw_dict.items() if k not in dedicated}

        return cls({**dedicated, **others})
