"""The instance for exporting data. (:mod:`qurry.qurrium.experiment.export`)"""

import os
from typing import Union, Any
from dataclasses import dataclass
from pathlib import Path

from ..json_io import WrittenContentType
from ...capsule import quickJSON, DEFAULT_ENCODING, DEFAULT_INDENT, DEFAULT_MODE


@dataclass(frozen=True)
class Export:
    """Data-stored namedtuple with all experiments data which is jsonable.

    ### Single experiment:

    For the :meth:`write` function actually exports 5 different files
    respecting to `args`, `advent`, `legacy`, `tales`, and `myths` like:

    .. code-block:: python

        files = {
            'folder': './bla_exp/',
            'qurryinfo': './bla_exp/qurryinfo.json',
            'args': './bla_exp/args/bla_exp.id={exp_id}.args.json',
            'advent': './bla_exp/advent/bla_exp.id={exp_id}.advent.json',
            'legacy': './bla_exp/legacy/bla_exp.id={exp_id}.legacy.json',
            'tales': './bla_exp/tales/bla_exp.id={exp_id}.tales.json',
            'myths': './bla_exp/myths/bla_exp.id={exp_id}.myths.json',
        }

    which `bla_exp` is the example filename.

    ### Multi-experiment:

    If this experiment is called by
    :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`,
    then the it will be named after `summoner_name` as known as the name of
    :class:`~qurry.qurrium.multimanager.multimanager.MultiManager`.

    .. code-block:: python

        files = {
            'folder': './BLABLA_project/',
            'qurryinfo': './BLABLA_project/qurryinfo.json',
            'args': './BLABLA_project/args/index={serial}.id={exp_id}.args.json',
            'advent': './BLABLA_project/advent/index={serial}.id={exp_id}.advent.json',
            'legacy': './BLABLA_project/legacy/index={serial}.id={exp_id}.legacy.json',
            'tales': './BLABLA_project/tales/index={serial}.id={exp_id}.tales.json',
            'myths':
                './BLABLA_project/myths/index={serial}.id={exp_id}.myths.json',
        }

    which `BLBLA_project` is the example
    :class:`~qurry.qurrium.multimanager.multimanager.MultiManager` name
    stored at `summoner_name` in
    :class:`~qurry.qurrium.experiment.arguments.Commonparams.summoner_name`.
    At this senerio, the `exp_name` will never apply as filename.
    """

    exp_id: str
    """The id of experiment used in filenames."""
    identifier: str
    """The identifier of experiment used in filenames."""
    folder: str
    """The folder of experiment."""
    save_location: Union[Path, str]
    """The save location of experiment used in filenames."""

    folder_filenames_writtens: list[tuple[str, str, WrittenContentType[Any]]]
    """The written contents of experiment used in exporting."""

    def __post_init__(self):
        invalid_filenames = {
            key: fname
            for key, fname, writtens in self.folder_filenames_writtens
            if any(k not in fname for k in [self.identifier, self.exp_id])
        }
        if invalid_filenames:
            raise ValueError(
                "The identifier or exp_id must be included in all filenames. "
                + f"Invalid filenames: {invalid_filenames}"
            )
        invalid_written_contents = {
            key: writtens.keys()
            for key, fname, writtens in self.folder_filenames_writtens
            if "files" in writtens
        }
        if invalid_written_contents:
            raise ValueError(
                "The written_contents must not contain 'files' key, "
                + "which is reserved for internal use. "
                + f"Invalid written_contents: {invalid_written_contents.keys()}"
            )

    def write(self) -> tuple[str, dict[str, str]]:
        """Export the experiment data, if there is a previous export, then will overwrite.

        Returns:
            tuple[str, dict[str, str]]:
                The first element is the id of experiment,
                the second element is the dictionary of files of experiment.
        """

        folder_path = Path(self.folder)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        files = {
            "folder": folder_path,
            "qurryinfo": folder_path / "qurryinfo.json",
        }
        for key, fname, writtens in self.folder_filenames_writtens:
            files[key] = folder_path / key / fname
            if not os.path.exists(folder_path / key):
                os.mkdir(folder_path / key)
        files_str = {k: str(v) for k, v in files.items()}

        for key, fname, writtens in self.folder_filenames_writtens:
            writtens.update({"files": files_str})
            quickJSON(
                content=writtens,
                filename=files[key],
                mode=DEFAULT_MODE,
                indent=DEFAULT_INDENT,
                encoding=DEFAULT_ENCODING,
                save_location=self.save_location,
            )

        return self.exp_id, files_str
