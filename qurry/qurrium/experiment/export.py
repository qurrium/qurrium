"""The module of exporting the experiment data. (:mod:`qurry.qurrium.container.export`)"""

import os
from typing import Optional, NamedTuple, Union, Any
from collections.abc import Hashable
from pathlib import Path
import warnings
import gc
import tqdm

from .arguments import CommonparamsDict, REQUIRED_FOLDER
from ...tools import ParallelManager
from ...capsule import quickJSON


class Export(NamedTuple):
    """Data-stored namedtuple with all experiments data which is jsonable."""

    exp_id: str
    """ID of experiment, which will be packed into `.args.json`."""
    exp_name: str
    """Name of the experiment, which will be packed into `.args.json`. 
    If this experiment is called by multimanager, 
    then this name will never apply as filename."""
    # Arguments for multi-experiment
    serial: Optional[int]
    """Index of experiment in a multiOutput, which will be packed into `.args.json`."""
    summoner_id: Optional[str]
    """ID of experiment of the multiManager, which will be packed into `.args.json`."""
    summoner_name: Optional[str]
    """Name of experiment of the multiManager, which will be packed into `.args.json`."""

    filename: str
    """The name of file to be exported, 
    it will be decided by the :meth:`.export` when it's called.
    More info in the pydoc of :prop:`files` or :meth:`.export`, 
    which will be packed into `.args.json`.
    """
    files: dict[str, str]
    """The list of file to be exported.

    ### Single experiment:

    For the :meth:`.write` function actually exports 4 different files
    respecting to `adventure`, `legacy`, `tales`, and `reports` like:

    .. code-block:: python
        files = {
            'folder': './bla_exp/',
            'qurryinfo': './bla_exp/qurryinfo.json',

            'args': './bla_exp/args/bla_exp.id={exp_id}.args.json',
            'advent': './bla_exp/advent/bla_exp.id={exp_id}.advent.json',
            'legacy': './bla_exp/legacy/bla_exp.id={exp_id}.legacy.json',
            'tales.dummyx1': './bla_exp/tales/bla_exp.id={exp_id}.dummyx1.json',
            'tales.dummyx2': './bla_exp/tales/bla_exp.id={exp_id}.dummyx2.json',
            ...
            'tales.dummyxn': './bla_exp/tales/bla_exp.id={exp_id}.dummyxn.json',
            'reports': './bla_exp/reports/bla_exp.id={exp_id}.reports.json',
            'reports.tales.dummyz1': './bla_exp/tales/bla_exp.id={exp_id}.dummyz1.reports.json',
            'reports.tales.dummyz2': './bla_exp/tales/bla_exp.id={exp_id}.dummyz2.reports.json',
            ...
            'reports.tales.dummyzm': './bla_exp/tales/bla_exp.id={exp_id}.dummyzm.reports.json',
        }

    which `bla_exp` is the example filename.

    ### Multi-experiment:

    If this experiment is called by :cls:`MultiManager`, 
    then the it will be named after `summoner_name` as known as the name of :cls:`MultiManager`.

    .. code-block:: python
        files = {
            'folder': './BLABLA_project/',
            'qurryinfo': './BLABLA_project/qurryinfo.json',

            'args': './BLABLA_project/args/index={serial}.id={exp_id}.args.json',
            'advent': './BLABLA_project/advent/index={serial}.id={exp_id}.advent.json',
            'legacy': './BLABLA_project/legacy/index={serial}.id={exp_id}.legacy.json',
            'tales.dummyx1': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyx1.json',
            'tales.dummyx2': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyx2.json',
            ...
            'tales.dummyxn': './BLABLA_project/tales/index={serial}.id={exp_id}.dummyxn.json',
            'reports': './BLABLA_project/reports/index={serial}.id={exp_id}.reports.json',
            'reports.tales.dummyz1': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyz1.reports.json',
            'reports.tales.dummyz2': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyz2.reports.json',
            ...
            'reports.tales.dummyzm': 
                './BLABLA_project/tales/index={serial}.id={exp_id}.dummyzm.reports.json',
        }

    which `BLBLA_project` is the example :cls:`MultiManager` name 
    stored at :prop:`commonparams.summoner_name`.
    At this senerio, the :prop:`exp_name` will never apply as filename.

    """

    args: dict[str, Any]
    """Construct the experiment's parameters, which will be packed into `.args.json`."""
    commons: CommonparamsDict
    """Construct the experiment's common parameters, which will be packed into `.args.json`."""
    outfields: dict[str, Any]
    """Recording the data of other unused arguments, which will be packed into `.args.json`."""

    adventures: dict[str, Any]
    """Recording the data of 'beforeward', which will be packed into `.advent.json`. 
    *~ A Great Adventure begins ~*"""
    legacy: dict[str, Any]
    """Recording the data of 'afterward', which will be packed into `.legacy.json`. 
    *~ The Legacy remains from the achievement of ancestors ~*"""
    tales: dict[str, Any]
    """Recording the data of 'side_product' in 'afterward' and 'beforewards' for API, 
    which will be packed into `.*.tales.json`. 
    *~ Tales of braves circulate ~*"""

    reports: dict[Hashable, dict[str, Any]]
    """Recording the data of 'reports', which will be packed into `.reports.json`. 

    ### Reports format:

    .. code-block:: python
        reports = {
            1: { ...quantities, 'input': { ... }, 'header': { ... }, },
            2: { ...quantities, 'input': { ... }, 'header': { ... }, },
            ...
            {serial}: { ...quantities, 'input': { ... }, 'header': { ... }, },
        }

    *~ The guild concludes the results. ~*"""
    tales_reports: dict[str, dict[Hashable, dict[str, Any]]]
    """Recording the data of 'side_product' in 'reports' for API, 
    which will be packed into `.*.reprts.json`. 

    ### Tales Reports format:

    .. code-block:: python
        tales_reports = {
            'dummyz1': {
                1: { ... },
                2: { ... },
                ...
                {serial}: { ... },
            },
            'dummyz2': {
                1: { ... },
                2: { ... },
                ...
                {serial}: { ... },
            },
            ...
            'dummyz': {
                1: { ... },
                2: { ... },
                ...
                {serial}: { ... },
            },
        }

    *~ Tales of braves circulate ~*"""

    def write(
        self,
        mode: str = "w+",
        indent: int = 2,
        encoding: str = "utf-8",
        jsonable: bool = False,
        mute: bool = False,
        multiprocess: bool = False,
        pbar: Optional[tqdm.tqdm] = None,
    ) -> tuple[str, dict[str, str]]:
        """Export the experiment data, if there is a previous export, then will overwrite.

        Args:
            save_location (Optional[Union[Path, str]], optional):
                Where to save the export content as `json` file.
                If `save_location == None`, then use the value in `self.commons` to be exported,
                if it's None too, then raise error.
                Defaults to `None`.

            mode (str):
                Mode for :func:`open` function, for :func:`mori.quickJSON`. Defaults to 'w+'.
            indent (int, optional):
                Indent length for json, for :func:`mori.quickJSON`. Defaults to 2.
            encoding (str, optional):
                Encoding method, for :func:`mori.quickJSON`. Defaults to 'utf-8'.
            jsonable (bool, optional):
                Whether to transpile all object to jsonable via :func:`mori.jsonablize`,
                for :func:`mori.quickJSON`. Defaults to False.
            mute (bool, optional):
                Whether to mute the output, for :func:`mori.quickJSON`. Defaults to False.
            multiprocess (bool, optional):
                Whether to use multiprocess to export, Defaults to False.
                It's dangerous to use multiprocess to export. It may cause memory leak.
            pbar (Optional[tqdm.tqdm], optional):
                The progress bar for exporting. Defaults to None.

        Returns:
            tuple[str, dict[str, str]]:
                The first element is the id of experiment,
                the second element is the dictionary of files of experiment.
        """

        export_set: dict[
            str,
            Union[
                dict[str, Any],
                list[Any],
                tuple[Any, ...],
                dict[Hashable, dict[str, Any]],
            ],
        ] = {}
        # args ...............  # arguments, commonparams, outfields, files
        export_set["args"] = {
            "arguments": self.args,
            "commonparams": self.commons,
            "outfields": self.outfields,
            "files": self.files,
        }
        # advent .............  # adventures
        export_set["advent"] = {
            "files": self.files,
            "adventures": self.adventures,
        }
        # legacy .............  # legacy
        export_set["legacy"] = {
            "files": self.files,
            "legacy": self.legacy,
        }
        # tales ..............  # tales
        for tk, tv in self.tales.items():
            if isinstance(tv, (dict, list, tuple)):
                export_set[f"tales.{tk}"] = tv
            else:
                export_set[f"tales.{tk}"] = [tv]
            if f"tales.{tk}" not in self.files:
                warnings.warn(f"tales.{tk} is not in export_names, it's not exported.")
        # reports ............  # reports
        export_set["reports"] = {
            "files": self.files,
            "reports": self.reports,
        }
        # reports.tales ......  # tales_reports
        for tk, tv in self.tales_reports.items():
            if isinstance(tv, (dict, list, tuple)):
                export_set[f"reports.tales.{tk}"] = tv
            else:
                export_set[f"reports.tales.{tk}"] = [tv]
            if f"reports.tales.{tk}" not in self.files:
                warnings.warn(f"reports.tales.{tk} is not in export_names, it's not exported.")
        # Exportation
        if pbar is not None:
            pbar.set_description_str(
                "Exporting "
                + (f"{self.summoner_name}/" if self.summoner_name else "")
                + f"{self.exp_name}..."
            )
        folder = Path(self.commons["save_location"]) / Path(self.files["folder"])
        if not os.path.exists(folder):
            os.mkdir(folder)
        for k in REQUIRED_FOLDER:
            if not os.path.exists(folder / k):
                os.mkdir(folder / k)

        if multiprocess:
            pool = ParallelManager()
            pool.starmap(
                quickJSON,
                [
                    (
                        content,
                        str(Path(self.commons["save_location"]) / self.files[filekey]),
                        mode,
                        indent,
                        encoding,
                        jsonable,
                        Path("./"),
                        mute,
                    )
                    for filekey, content in export_set.items()
                ],
            )
        else:
            for filekey, content in export_set.items():
                quickJSON(
                    content=content,
                    filename=str(Path(self.commons["save_location"]) / self.files[filekey]),
                    mode=mode,
                    indent=indent,
                    encoding=encoding,
                    jsonable=jsonable,
                    save_location=Path("./"),
                    mute=mute,
                )

        del export_set
        gc.collect()
        return self.exp_id, self.files
