"""Chunk distribution for multiprocess. (:mod:`qurry.qurrium.utils.chunk`)

This module provides functions to calculate chunk size and distribute tasks
"""

from ...tools.parallelmanager import DEFAULT_POOL_SIZE, CPU_COUNT


def very_easy_chunk_size(
    tasks_num: int,
    num_process: int = DEFAULT_POOL_SIZE,
    max_chunk_size: int = CPU_COUNT * 4,
) -> int:
    """Calculate the chunk size for multiprocess.

    Args:
        tasks_num (int): The number of tasks.
        num_process (int, optional):
            The chunk size. Defaults to CPU_COUNT * 4.
        max_chunk_size (int, optional):
            The maximum chunk size. Defaults to CPU_COUNT * 4.

    Returns:
        int: The chunk size.
    """
    chunks_num = int(tasks_num / num_process) + 1
    while chunks_num > max_chunk_size:
        num_process *= 2
        chunks_num = int(tasks_num / num_process) + 1
    return chunks_num


def very_easy_chunk_distribution(
    respect_memory_array: list[tuple[str, int]],
    num_process: int = DEFAULT_POOL_SIZE,
    max_chunk_size: int = CPU_COUNT * 4,
) -> tuple[int, list[tuple[str, int]], list[list[str]]]:
    """Distribute the chunk for multiprocess.
    The chunk distribution is based on the number of CPU cores.

    Args:
        respect_memory_array (list[tuple[str, int]]):
            The array of respect memory.
            Each element is a tuple of (id, memory).
            The id is the ID of the experiment, and the memory is the memory usage.
            The array is sorted by the memory usage.
        num_process (int, optional):
            The chunk size. Defaults to CPU_COUNT * 4.

    Returns:
        tuple[int, list[tuple[str, int]], list[list[str]]]:
            The chunk distribution is a list of tuples of (id, memory).
    """

    chunks_num = int(len(respect_memory_array) / num_process) + 1
    while chunks_num > max_chunk_size:
        num_process *= 2
        chunks_num = int(len(respect_memory_array) / num_process) + 1
    chunks_sorted_list = []
    distributions = [[] for _ in range(num_process)]

    for i in range(num_process):
        for j in range(chunks_num):
            # Distribute the chunks in a round-robin fashion
            idx = j * num_process + i if j % 2 == 0 else (j + 1) * num_process - i - 1
            if idx < len(respect_memory_array):
                chunks_sorted_list.append(respect_memory_array[idx])
                distributions[i].append(idx)

    return chunks_num, chunks_sorted_list, distributions
