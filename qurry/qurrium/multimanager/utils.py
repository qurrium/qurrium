"""MultiManager Utilities (:mod:`qurry.qurrium.multimanager.utils`)"""

from multiprocessing import get_context

from .arguments import MultiCommonparams
from .beforewards import Before
from .process import multiprocess_exporter_wrapper
from ..utils.chunk import very_easy_chunk_distribution
from ..container import ExperimentContainer, _E
from ...tools import qurry_progressbar, DEFAULT_POOL_SIZE
from ...capsule import quickJSON


def experiment_writer(
    experiment_container: ExperimentContainer[_E],
    beforewards: Before,
    multicommons: MultiCommonparams,
    taglist_name: str,
    indent: int = 2,
    encoding: str = "utf-8",
    export_transpiled_circuit: bool = False,
    multiprocess: bool = False,
):
    """Write the experiment.

    Args:
        experiment_container (ExperimentContainer[_E]):
            The container of the experiment.
        beforewards (Before):
            The beforewards of the experiment.
        multicommons (MultiCommonparams):
            The common parameters of the experiment.
        taglist_name (str):
            The name of the taglist.
        indent (int, optional):
            The indent of the json file. Defaults to 2.
        encoding (str, optional):
            The encoding of the json file. Defaults to "utf-8".
        export_transpiled_circuit (bool, optional):
            Whether to export the transpiled circuit. Defaults to False.
        multiprocess (bool, optional):
            Whether to use multiprocess. Defaults to False.
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

        first_export = experiment_container[respect_memory_array[0][0]].write(
            save_location=multicommons.save_location,
            mode="w+",
            indent=indent,
            encoding=encoding,
            jsonable=True,
            export_transpiled_circuit=export_transpiled_circuit,
            qurryinfo_hold_access=multicommons.summoner_id,
            pbar=None,
        )

        chunks_num, chunks_sorted_list, _distributions = very_easy_chunk_distribution(
            respect_memory_array[1:], DEFAULT_POOL_SIZE, DEFAULT_POOL_SIZE * 2
        )

        exporting_pool = get_context("spawn").Pool(
            processes=DEFAULT_POOL_SIZE, maxtasksperchild=chunks_num * 2
        )
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
                            "w+",
                            indent,
                            encoding,
                            True,
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

        all_qurryinfo[first_export[0]] = first_export[1]
        all_qurryinfo = dict(sorted(all_qurryinfo.items(), key=lambda x: exps_serial[x[0]]))

    else:
        all_qurryinfo = {}
        single_exporting_progress = qurry_progressbar(
            beforewards.exps_config,
            desc="Exporting experiments...",
            bar_format="qurry-barless",
        )
        for id_exec in single_exporting_progress:
            tmp_id, tmp_qurryinfo_content = experiment_container[id_exec].write(
                save_location=multicommons.save_location,
                mode="w+",
                indent=indent,
                encoding=encoding,
                jsonable=True,
                qurryinfo_hold_access=multicommons.summoner_id,
                export_transpiled_circuit=export_transpiled_circuit,
                multiprocess=True,
                pbar=single_exporting_progress,
            )
            assert id_exec == tmp_id, "ID is not consistent."
            all_qurryinfo[id_exec] = tmp_qurryinfo_content

    # for id_exec, files in all_qurryinfo_items:
    for id_exec, files in qurry_progressbar(
        all_qurryinfo.items(),
        desc="Loading file infomation...",
        bar_format="qurry-barless",
    ):
        beforewards.files_taglist[experiment_container[id_exec].commons.tags].append(files)

    print("| Exporting file taglist...")
    beforewards.files_taglist.export(
        name=None,
        save_location=multicommons.export_location,
        taglist_name=f"{taglist_name}",
        filetype=multicommons.filetype,
        open_args={
            "mode": "w+",
            "encoding": encoding,
        },
        json_dump_args={
            "indent": indent,
        },
    )
    print(f"| Exporting {all_qurryinfo_loc}...")
    quickJSON(
        content=all_qurryinfo,
        filename=all_qurryinfo_loc,
        mode="w+",
        jsonable=True,
        indent=indent,
        encoding=encoding,
        mute=True,
    )
    del all_qurryinfo
    print(f"| Exporting {all_qurryinfo_loc} done.")
