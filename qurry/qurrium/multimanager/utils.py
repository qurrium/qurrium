"""MultiManager Utilities (:mod:`qurry.qurrium.multimanager.utils`)"""

from multiprocessing import get_context

from .arguments import MultiCommonparams
from .beforewards import Before
from .process import multiprocess_exporter_wrapper
from ..utils.iocontrol import RJUST_LEN, serial_naming
from ..utils.chunk import very_easy_chunk_distribution
from ..container import ExperimentContainer, _E, QuantityContainer
from ...tools import qurry_progressbar, DEFAULT_POOL_SIZE
from ...capsule import quickJSON, DEFAULT_MODE, DEFAULT_ENCODING, DEFAULT_INDENT


def experiment_writer(
    experiment_container: ExperimentContainer[_E],
    beforewards: Before,
    multicommons: MultiCommonparams,
    export_transpiled_circuit: bool = False,
    multiprocess: bool = False,
) -> dict[str, dict[str, str]]:
    """Write the experiment.

    Args:
        experiment_container (ExperimentContainer[_E]):
            The container of the experiment.
        beforewards (Before):
            The beforewards of the experiment.
        multicommons (MultiCommonparams):
            The common parameters of the experiment.
        export_transpiled_circuit (bool, optional):
            Whether to export the transpiled circuit. Defaults to False.
        multiprocess (bool, optional):
            Whether to use multiprocess. Defaults to False.

    Returns:
        The dictionary of the exported information of the experiment.
        The keys are the experiment IDs,
        and the values are the dictionaries of the exported information.
    """

    all_qurryinfo_loc = multicommons.export_location / "qurryinfo.json"

    if multiprocess:
        respect_memory_array = [
            (id_exec, int(experiment_container[id_exec].memory_usage_factor))
            for id_exec in beforewards.exps_config.keys()
        ]
        respect_memory_array.sort(key=lambda x: x[1])
        exps_serial = {
            id_exec: default_order for default_order, id_exec in enumerate(beforewards.exps_config)
        }

        tmp_export_info = experiment_container[respect_memory_array[0][0]].write(
            save_location=multicommons.save_location,
            export_transpiled_circuit=export_transpiled_circuit,
            qurryinfo_hold_access=multicommons.summoner_id,
            pbar=None,
        )

        chunks_num, chunks_sorted_list, _ = very_easy_chunk_distribution(
            respect_memory_array=respect_memory_array[1:],
            num_process=DEFAULT_POOL_SIZE,
            max_chunk_size=min(max(1, len(respect_memory_array[1:]) // DEFAULT_POOL_SIZE), 40),
        )

        exporting_pool = get_context("spawn").Pool(processes=DEFAULT_POOL_SIZE, maxtasksperchild=4)
        with exporting_pool as ep:
            export_imap_result = qurry_progressbar(
                ep.imap_unordered(
                    multiprocess_exporter_wrapper,
                    (
                        (
                            id_exec,
                            experiment_container[id_exec].export(
                                save_location=multicommons.save_location,
                                export_transpiled_circuit=export_transpiled_circuit,
                            ),
                        )
                        for id_exec, memory_usage in chunks_sorted_list
                    ),
                    chunksize=chunks_num,
                ),
                total=len(chunks_sorted_list),
                desc="Exporting experiments...",
                bar_format="qurry-barless",
            )
            all_qurryinfo = dict(export_imap_result)

        all_qurryinfo[tmp_export_info[0]] = tmp_export_info[1]
        all_qurryinfo = dict(sorted(all_qurryinfo.items(), key=lambda x: exps_serial[x[0]]))

    else:
        all_qurryinfo = {}
        single_exporting_progress = qurry_progressbar(
            beforewards.exps_config,
            desc="Exporting experiments...",
            bar_format="qurry-barless",
        )
        for id_exec in single_exporting_progress:
            tmp_export_info = experiment_container[id_exec].write(
                save_location=multicommons.save_location,
                qurryinfo_hold_access=multicommons.summoner_id,
                export_transpiled_circuit=export_transpiled_circuit,
                multiprocess=True,
                pbar=single_exporting_progress,
            )
            assert (
                id_exec == tmp_export_info[0]
            ), f"ID is not consistent: {id_exec} != {tmp_export_info[0]}."
            all_qurryinfo[id_exec] = tmp_export_info[1]

    # for id_exec, files in all_qurryinfo_items:
    print(f"| Exporting {all_qurryinfo_loc}...")
    quickJSON(
        content=all_qurryinfo,
        filename=all_qurryinfo_loc,
        mode=DEFAULT_MODE,
        jsonable=False,
        indent=DEFAULT_INDENT,
        encoding=DEFAULT_ENCODING,
    )
    print(f"| Exporting {all_qurryinfo_loc} done.")
    return all_qurryinfo


def multimanager_report_naming(
    analysis_name: str,
    no_serialize: bool,
    quantities_container: QuantityContainer,
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
