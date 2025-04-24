"""MultiManager Utilities (:mod:`qurry.qurrium.multimanager.utils`)"""

import gc
from multiprocessing import get_context

from .arguments import MultiCommonparams
from .beforewards import Before
from .process import (
    multiprocess_exporter,
    single_process_exporter,
    very_easy_chunk_distribution,
    multiprocess_exporter_wrapper,
)
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
        exps_serial = {id_exec: default for default, id_exec in enumerate(beforewards.exps_config)}
        first_export = multiprocess_exporter(
            id_exec=respect_memory_array[0][0],
            exps_export=experiment_container[respect_memory_array[0][0]].export(
                save_location=multicommons.save_location,
                export_transpiled_circuit=export_transpiled_circuit,
            ),
            mode="w+",
            indent=indent,
            encoding=encoding,
            jsonable=True,
            mute=True,
            pbar=None,
        )
        chunks_sorted_list = very_easy_chunk_distribution(
            respect_memory_array[1:], DEFAULT_POOL_SIZE * 2
        )

        exporting_pool = get_context("spawn").Pool(
            processes=DEFAULT_POOL_SIZE, maxtasksperchild=DEFAULT_POOL_SIZE
        )
        with exporting_pool as ep:
            export_imap_result = qurry_progressbar(
                ep.imap_unordered(
                    multiprocess_exporter_wrapper,
                    [
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
                            True,
                            None,
                        )
                        for id_exec, memory_usage in chunks_sorted_list
                    ],
                    chunksize=DEFAULT_POOL_SIZE * 2,
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
        exps_export_progress = qurry_progressbar(
            beforewards.exps_config,
            desc="Exporting experiments...",
            bar_format="qurry-barless",
        )
        for id_exec in exps_export_progress:
            tmp_id, tmp_qurryinfo_content = single_process_exporter(
                id_exec=id_exec,
                exps_export=experiment_container[id_exec].export(
                    save_location=multicommons.save_location,
                    export_transpiled_circuit=export_transpiled_circuit,
                ),
                mode="w+",
                indent=indent,
                encoding=encoding,
                jsonable=True,
                mute=True,
                pbar=None,
            )
            assert id_exec == tmp_id, "ID is not consistent."
            all_qurryinfo[id_exec] = tmp_qurryinfo_content

    gc.collect()

    # for id_exec, files in all_qurryinfo_items:
    for id_exec, files in all_qurryinfo.items():
        beforewards.files_taglist[experiment_container[id_exec].commons.tags].append(files)
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

    quickJSON(
        content=all_qurryinfo,
        filename=all_qurryinfo_loc,
        mode="w+",
        jsonable=True,
        indent=indent,
        encoding=encoding,
        mute=True,
    )

    gc.collect()
