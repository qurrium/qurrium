"""Handle Bitstring Index. (:mod:`qurry.qurrium.utils.bitstring`)"""


def bitstring_mapping_getter(
    counts: list[dict[str, int]], registers_mapping: dict[int, int]
) -> tuple[dict[int, int], dict[int, int]]:
    """Get the bitstring mapping and the final mapping.

    When you have multiple classical registers clusters,
    the bitstring will be separated by space.
    So we need to shift the index of the `registers_mapping` to get the correct mapping,
    it usually happens when you measure a circuit with its own classical registers.

    For example, we consider a circuit with 4 qubits and
    its own classical registers cluster of 4 bits.
    We want to measure the last 2 qubits,
    then this will add a new classical register with 2 bits.
    Here is the example of the counts:

    .. code-block:: python

        {
            "00 0001": 75,
            "01 0001": 171,
            "11 0001": 584,
            "10 0001": 194
        }

    Qurrium will keep another mapping information on the classical register we added.

    .. code-block:: python

        {
            "registers_mapping": {
                2: 0,
                3: 1
            }
        }

    Everything looks good, right?

    The post-processing of Qurrium will read the bitstring from right to left,
    to handle the little-endian format of Qiskit.
    If we use the `registers_mapping` directly,
    it will map to the wrong cluster of classical registers,
    which is the first 2 bits of first cluster of 4 bits in this example,
    where the results is "01" in "0001".

    So we need to shift the index of the `registers_mapping` by the total of
    the length of the other clusters of classical registers and the space.
    In this example, the shift is `4 (the length of the first cluster) + 1 (the space) = 5`.
    So the final mapping should be

    .. code-block:: python

        {
            "bitstring_mapping": {
                0: 5,
                1: 6,
            },  # original index to shifted index
            "final_mapping": {
                2: 5,
                3: 6
            }, # qubit index to shifted index
        }

    The `bitstring_mapping` is the mapping of the original index to the shifted index.
    It is useful when you want to know the mapping of all classical registers.

    Args:
        counts (list[dict[str, int]]):
            The counts of the experiment.
        registers_mapping (dict[int, int]):
            The mapping of the index of selected qubits to the index of the classical register.

    Returns:
        tuple[dict[int, int], dict[int, int]]: The bitstring mapping and the final mapping.
    """

    bitstring_sampling = next(iter(counts[0].keys()))
    bitstring_sampling_divided = bitstring_sampling.split(" ")

    if len(bitstring_sampling_divided) > 1:
        bitstring_shift = len(bitstring_sampling_divided) - 1
        for clbit_cluster in bitstring_sampling_divided[1:]:
            bitstring_shift += len(clbit_cluster)
        bitstring_mapping = {v: v + bitstring_shift for v in registers_mapping.values()}
        final_mapping = {k: bitstring_mapping[v] for k, v in registers_mapping.items()}
        return bitstring_mapping, final_mapping

    return {v: v for v in registers_mapping.values()}, registers_mapping
