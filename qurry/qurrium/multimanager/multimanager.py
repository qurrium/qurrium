"""MultiManager - The manager of multiple experiments. (:mod:`qurry.qurry.qurrium.multimanager`)"""

import os
import gc
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Union, Optional, Any, Type, Generic
from collections.abc import Hashable
from uuid import uuid4

from qiskit.providers import Backend

from .arguments import MultiCommonparams, PendingStrategyLiteral, PendingTargetProviderLiteral
from .beforewards import Before
from .afterwards import After
from .process import (
    datetimedict_process,
    multiprocess_exporter,
    multiprocess_builder,
    single_process_exporter,
)
from ..container import ExperimentContainer, QuantityContainer, _E
from ..utils.iocontrol import naming, RJUST_LEN, IOComplex
from ...tools import qurry_progressbar, ParallelManager, GeneralSimulator, DatetimeDict
from ...capsule import quickJSON
from ...capsule.mori import TagList, GitSyncControl
from ...declare import BaseRunArgs, AnalyzeArgs
from ...exceptions import (
    QurryProtectContent,
    QurryResetAccomplished,
    QurryResetSecurityActivated,
)


class MultiManager(Generic[_E]):
    """The manager of multiple experiments."""

    __name__ = "MultiManager"

    multicommons: MultiCommonparams
    beforewards: Before
    afterwards: After

    quantity_container: QuantityContainer
    """The container of quantity."""
    exps: ExperimentContainer[_E]
    """The experiments container."""

    _unexports: list[str] = ["retrievedResult"]
    """The content would not be exported."""
    _not_sync = ["allCounts", "retrievedResult"]
    """The content would not be synchronized."""
    after_lock: bool = False
    """Protect the :cls:`afterward` content to be overwritten. 
    When setitem is called and completed, it will be setted as `False` automatically.
    """
    mute_auto_lock: bool = False
    """Whether mute the auto-lock message."""

    def reset_afterwards(
        self,
        *args,
        security: bool = False,
        mute_warning: bool = False,
    ) -> None:
        """Reset the measurement and release memory for overwrite.

        Args:
            security (bool, optional): Security for reset. Defaults to `False`.
            mute_warning (bool, optional): Mute warning. Defaults to `False`.
        """

        if len(args) > 0:
            raise ValueError("Use '.reset(security=True)' to reset the afterwards.")

        if security and isinstance(security, bool):
            self.afterwards = self.afterwards._replace(retrievedResult=TagList(), allCounts={})
            gc.collect()
            if not mute_warning:
                warnings.warn("Afterwards reset accomplished.", QurryResetAccomplished)
        else:
            warnings.warn(
                "Reset does not execute to prevent executing accidentally, "
                + "if you are sure to do this, then use '.reset(security=True)'.",
                QurryResetSecurityActivated,
            )

    def unlock_afterward(self, mute_auto_lock: bool = False):
        """Unlock the :cls:`afterward` content to be overwritten.

        Args:
            mute_auto_lock (bool, optional):
                Mute anto-locked message for the unlock of this time. Defaults to False.
        """
        self.after_lock = True
        self.mute_auto_lock = mute_auto_lock

    def __setitem__(self, key, value) -> None:
        if key in self.beforewards._fields:
            self.beforewards = self.beforewards._replace(**{key: value})

        elif key in self.afterwards._fields:
            if self.after_lock and isinstance(self.after_lock, bool):
                self.afterwards = self.afterwards._replace(**{key: value})
            else:
                raise QurryProtectContent(
                    f"Can't set value to :cls:`afterward` field {key} "
                    + "because it's locked, use `.unlock_afterward()` "
                    + "to unlock before setting item ."
                )

        else:
            raise ValueError(
                f"{key} is not a valid field of '{Before.__name__}' and '{After.__name__}'."
            )

        gc.collect()
        if self.after_lock is not False:
            self.after_lock = False
            if not self.mute_auto_lock:
                print(
                    "after_lock is locked automatically now, "
                    + "you can unlock by using `.unlock_afterward()` "
                    + "to set value to :cls:`afterward`."
                )
            self.mute_auto_lock = False

    def __getitem__(self, key) -> Any:
        if key in self.beforewards._fields:
            return getattr(self.beforewards, key)
        if key in self.afterwards._fields:
            return getattr(self.afterwards, key)
        raise ValueError(
            f"{key} is not a valid field of '{Before.__name__}' and '{After.__name__}'."
        )

    @property
    def summoner_id(self) -> str:
        """ID of experiment of the MultiManager."""
        return self.multicommons.summoner_id

    @property
    def id(self) -> str:
        """ID of experiment of the MultiManager."""
        return self.multicommons.summoner_id

    @property
    def summoner_name(self) -> str:
        """Name of experiment of the MultiManager."""
        return self.multicommons.summoner_name

    @property
    def name(self) -> str:
        """Name of experiment of the MultiManager."""
        return self.multicommons.summoner_name

    def __init__(
        self,
        naming_complex: IOComplex,
        multicommons: MultiCommonparams,
        beforewards: Before,
        afterwards: After,
        quantity_container: QuantityContainer,
        outfields: dict[str, Any],
        gitignore: Optional[Union[GitSyncControl, list[str]]] = None,
    ):
        """Initialize the multi-experiment."""

        if gitignore is None:
            self.gitignore = GitSyncControl()
        elif isinstance(gitignore, list):
            self.gitignore = GitSyncControl(gitignore)
        elif isinstance(gitignore, GitSyncControl):
            self.gitignore = gitignore
        else:
            raise ValueError(f"gitignore must be list or GitSyncControl, not {type(gitignore)}.")

        self.exps = ExperimentContainer()

        self.naming_complex = naming_complex
        self.multicommons = multicommons
        self.beforewards = beforewards
        self.afterwards = afterwards
        self.quantity_container = quantity_container
        self.outfields = outfields

    def __repr__(self):
        return (
            f"<{self.__name__}("
            + f'id="{self.multicommons.summoner_id}", '
            + f'name="{self.multicommons.summoner_name}", '
            + f"tags={self.multicommons.tags}, "
            + f'jobstype="{self.multicommons.jobstype}", '
            + f'pending_strategy="{self.multicommons.pending_strategy}", '
            + f"last_events={dict(self.multicommons.datetimes.last_events(3))}, "
            + f"exps_num={len(self.beforewards.exps_config)})>"
        )

    def _repr_oneline(self):
        return (
            f"<{self.__name__}("
            + f'id="{self.multicommons.summoner_id}", '
            + f'name="{self.multicommons.summoner_name}", '
            + f'jobstype="{self.multicommons.jobstype}", ..., '
            + f"exps_num={len(self.beforewards.exps_config)})>"
        )

    def _repr_oneline_no_id(self):
        return (
            f"<{self.__name__}("
            + f'name="{self.multicommons.summoner_name}", '
            + f'jobstype="{self.multicommons.jobstype}", ..., '
            + f"exps_num={len(self.beforewards.exps_config)})>"
        )

    def _repr_pretty_(self, p, cycle):
        max_events = 5

        if cycle:
            p.text(
                f"<{self.__name__}("
                + f'id="{self.multicommons.summoner_id}", '
                + f'name="{self.multicommons.summoner_name}", '
                + f'jobstype="{self.multicommons.jobstype}", ..., '
                + f"exps_num={len(self.beforewards.exps_config)})>"
            )
        else:
            with p.group(2, f"<{type(self).__name__}(", ")>"):
                p.text(f'id="{self.multicommons.summoner_id}",')
                p.breakable()
                p.text(f'name="{self.multicommons.summoner_name}",')
                p.breakable()
                p.text(f"tags={self.multicommons.tags},")
                p.breakable()
                p.text(f'jobstype="{self.multicommons.jobstype}",')
                p.breakable()
                p.text(f'pending_strategy="{self.multicommons.pending_strategy}",')
                p.breakable()
                p.text("last_events={")
                if len(self.multicommons.datetimes) > max_events:
                    p.breakable()
                    p.text("  ...,")
                for k, v in self.multicommons.datetimes.last_events(max_events):
                    p.breakable()
                    p.text(f"  '{k}': '{v}',")
                p.text("},")
                p.breakable()
                p.text(f"exps_num={len(self.beforewards.exps_config)}")

    def register(
        self,
        current_id: str,
        config: dict[str, Any],
        exps_instance: _E,
    ):
        """Register the experiment to multimanager.

        Args:
            current_id (str): ID of experiment.
            config (dict[str, Any]): The config of experiment.
            exps_instance (ExperimentPrototype): The instance of experiment.
        """

        assert exps_instance.commons.exp_id == current_id, (
            f"ID is not consistent, exp_id: {exps_instance.commons.exp_id} and "
            + f"current_id: {current_id}."
        )
        self.beforewards.exps_config[current_id] = config
        self.beforewards.circuits_num[current_id] = len(exps_instance.beforewards.circuit)
        self.beforewards.job_taglist[exps_instance.commons.tags].append(current_id)
        assert isinstance(exps_instance.commons.serial, int), (
            f"Serial is not int, exp_id: {exps_instance.commons.exp_id} and "
            + f"serial: {exps_instance.commons.serial}."
            + "It should be int."
        )
        self.beforewards.index_taglist[exps_instance.commons.tags].append(
            exps_instance.commons.serial
        )

    @classmethod
    def build(
        cls,
        config_list: list[dict[str, Any]],
        experiment_instance: Type[_E],
        summoner_name: Optional[str] = None,
        shots: Optional[int] = None,
        backend: Backend = GeneralSimulator(),
        tags: Optional[tuple[str, ...]] = None,
        manager_run_args: Optional[Union[BaseRunArgs, dict[str, Any]]] = None,
        jobstype: PendingTargetProviderLiteral = "local",
        pending_strategy: PendingStrategyLiteral = "tags",
        # save parameters
        save_location: Union[Path, str] = Path("./"),
        skip_writing: bool = False,
        multiprocess_build: bool = False,
        multiprocess_write: bool = False,
    ) -> "MultiManager":
        """Build the multi-experiment.

        Args:
            config_list (list[dict[str, Any]]): The list of config of experiments.
            experiment_instance (ExperimentPrototype): The instance of experiment.
            summoner_name (Optional[str], optional): Name of experiment of the MultiManager.
                Defaults to None.
            shots (Optional[int], optional): The shots of experiments. Defaults to None.
            backend (Backend, optional): The backend of experiments. Defaults to GeneralSimulator().
            tags (Optional[tuple[str, ...]], optional): The tags of experiments. Defaults to None.
            manager_run_args (Optional[Union[BaseRunArgs, dict[str, Any]]], optional):
                The arguments of manager run. Defaults to None.
            jobstype (PendingTargetProviderLiteral, optional):
                The jobstype of experiments. Defaults to "local".
            pending_strategy (PendingStrategyLiteral, optional):
                The pending strategy of experiments. Defaults to "tags".
            save_location (Union[Path, str], optional):
                Location of saving experiment. Defaults to Path("./").
            skip_writing (bool, optional):
                Whether skip writing. Defaults to False.
            multiprocess_build (bool, optional):
                Whether use multiprocess for building. Defaults to False.
            multiprocess_write (bool, optional):
                Whether use multiprocess for writing. Defaults to False.

        Returns:
            MultiManager: The container of experiments and multi-experiment.
        """

        if summoner_name is None:
            summoner_name = "multiexps"
        if tags is None:
            tags = ()
        if manager_run_args is None:
            manager_run_args = {}

        naming_complex = naming(
            exps_name=summoner_name,
            save_location=save_location,
        )

        multicommons, outfields = MultiCommonparams.build(
            {
                "summoner_id": str(uuid4()),
                "summoner_name": naming_complex.expsName,
                "tags": tags,
                "shots": shots,
                "backend": backend,
                "save_location": naming_complex.save_location,
                "export_location": naming_complex.export_location,
                "files": {},
                "jobstype": jobstype,
                "pending_strategy": pending_strategy,
                "manager_run_args": manager_run_args,
                "filetype": "json",
                "datetimes": DatetimeDict(),
                "outfields": {},
            }
        )

        assert (
            naming_complex.save_location == multicommons.save_location
        ), "| save_location is not consistent with namingCpx.save_location."

        current_multimanager = cls(
            naming_complex=naming_complex,
            multicommons=multicommons,
            beforewards=Before(
                exps_config={},
                circuits_num={},
                circuits_map=TagList(),
                pending_pool=TagList(),
                job_id=[],
                job_taglist=TagList(),
                files_taglist=TagList(),
                index_taglist=TagList(),
            ),
            afterwards=After(
                retrievedResult=TagList(),
                allCounts={},
            ),
            quantity_container=QuantityContainer(),
            outfields=outfields,
        )

        initial_config_list: list[dict[str, Any]] = []
        for serial, config in enumerate(config_list):
            config.pop("export", None)
            config.pop("pbar", None)
            config.pop("multiprocess", None)
            initial_config_list.append(
                {
                    **config,
                    "shots": config.get("shots", shots),
                    "backend": backend,
                    "exp_name": current_multimanager.multicommons.summoner_name,
                    "save_location": current_multimanager.multicommons.save_location,
                    "serial": serial,
                    "summoner_id": current_multimanager.multicommons.summoner_id,
                    "summoner_name": current_multimanager.multicommons.summoner_name,
                }
            )

        # initial_config_list_progress = qurry_progressbar(initial_config_list)
        # initial_config_list_progress.set_description_str("MultiManager building...")

        if multiprocess_build:
            pool = ParallelManager()
            exps_list: list[tuple[_E, dict[str, Any]]] = pool.process_map(
                multiprocess_builder,
                [
                    (
                        experiment_instance,
                        config,
                    )
                    for config in initial_config_list
                ],
                desc="MultiManager building...",
            )
        else:
            initial_config_list_progress = qurry_progressbar(
                initial_config_list,
                desc="MultiManager building...",
            )
            exps_list: list[tuple[_E, dict[str, Any]]] = [
                (experiment_instance.build(multiprocess=True, **config), config)
                for config in initial_config_list_progress
            ]

        for new_exps, config in exps_list:
            current_multimanager.register(
                current_id=new_exps.commons.exp_id,
                config=config,
                exps_instance=new_exps,
            )
            current_multimanager.exps[new_exps.commons.exp_id] = new_exps

        if not skip_writing:
            current_multimanager.write(multiprocess=multiprocess_write)

        return current_multimanager

    @classmethod
    def read(
        cls,
        summoner_name: str,
        experiment_instance: Type[_E],
        save_location: Union[Path, str] = Path("./"),
        is_read_or_retrieve: bool = False,
        read_from_tarfile: bool = False,
        encoding: str = "utf-8",
    ) -> "MultiManager":
        """Read the multi-experiment.

        Args:
            experiment_instance (ExperimentPrototype): The instance of experiment.
            summoner_name (Optional[str], optional): Name of experiment of the MultiManager.
                Defaults to None.
            save_location (Union[Path, str], optional):
                Location of saving experiment. Defaults to Path("./").
            is_read_or_retrieve (bool, optional): Whether read or retrieve. Defaults to False.
            read_from_tarfile (bool, optional): Whether read from tarfile. Defaults to False.

        Returns:
            MultiManager: The container of experiments and multi-experiment.
        """
        naming_complex = naming(
            is_read=is_read_or_retrieve,
            exps_name=summoner_name,
            save_location=save_location,
        )
        gitignore = GitSyncControl()
        gitignore.read(naming_complex.export_location)

        multiconfig_name_v5 = (
            naming_complex.export_location / f"{naming_complex.expsName}.multiConfig.json"
        )
        multiconfig_name_v7 = naming_complex.export_location / "multi.config.json"

        if naming_complex.tarLocation.exists():
            print(
                f"| Found the tarfile '{naming_complex.tarName}' "
                + f"in '{naming_complex.save_location}', decompressing is available."
            )
            if (not multiconfig_name_v5.exists()) and (not multiconfig_name_v7.exists()):
                print(
                    "| No multi.config file found, "
                    + f"decompressing all files in the tarfile '{naming_complex.tarName}'."
                )
                cls.easydecompress(naming_complex)
            elif read_from_tarfile:
                print(
                    f"| Decompressing all files in the tarfile '{naming_complex.tarName}'"
                    + f", replace all files in '{naming_complex.export_location}'."
                )
                cls.easydecompress(naming_complex)

        if multiconfig_name_v5.exists():
            print("| Found the multiConfig.json, reading in 'v5' file structure.")
            raw_multiconfig = MultiCommonparams.rawread(
                mutlticonfig_name=multiconfig_name_v5,
                save_location=naming_complex.save_location,
                export_location=naming_complex.export_location,
                encoding=encoding,
            )
            files: dict[str, Union[str, dict[str, str]]] = raw_multiconfig["files"]
            old_files = raw_multiconfig["files"].copy()
            beforewards = Before.read(
                export_location=naming_complex.export_location,
                file_location=files,
                version="v5",
            )
            afterwards = After.read(
                export_location=naming_complex.export_location,
                file_location=files,
                version="v5",
            )
            quantity_container = QuantityContainer()
            assert isinstance(files["tagMapQuantity"], dict), "Quantity must be dict."
            for qk in files["tagMapQuantity"].keys():
                quantity_container.read(
                    key=qk,
                    name=f"{naming_complex.expsName}.{qk}",
                    save_location=naming_complex.export_location,
                    version="v5",
                )

        elif multiconfig_name_v7.exists():
            raw_multiconfig = MultiCommonparams.rawread(
                mutlticonfig_name=multiconfig_name_v7,
                save_location=naming_complex.save_location,
                export_location=naming_complex.export_location,
                encoding=encoding,
            )
            files = raw_multiconfig["files"]
            old_files = {}
            beforewards = Before.read(export_location=naming_complex.export_location, version="v7")
            afterwards = After.read(export_location=naming_complex.export_location, version="v7")
            quantity_container = QuantityContainer()
            assert isinstance(files["quantity"], dict), "Quantity must be dict."
            for qk in files["quantity"].keys():
                quantity_container.read(
                    key=qk,
                    save_location=naming_complex.export_location,
                    name=f"{qk}",
                )
        else:
            print(f"| v5: {multiconfig_name_v5}")
            print(f"| v7: {multiconfig_name_v7}")
            raise FileNotFoundError(
                f"Can't find the multi.config file in '{naming_complex.expsName}'."
            )

        multicommons, outfields = MultiCommonparams.build(raw_multiconfig)

        datetimedict_process(
            multicommons=multicommons,
            naming_complex=naming_complex,
            multiconfig_name_v5=multiconfig_name_v5,
            multiconfig_name_v7=multiconfig_name_v7,
            is_read_or_retrieve=is_read_or_retrieve,
            read_from_tarfile=read_from_tarfile,
            old_files=old_files,
        )

        assert (
            naming_complex.save_location == multicommons.save_location
        ), "| save_location is not consistent with namingCpx.save_location."

        current_multimanager = cls(
            naming_complex=naming_complex,
            multicommons=multicommons,
            beforewards=beforewards,
            afterwards=afterwards,
            quantity_container=quantity_container,
            outfields=outfields,
            gitignore=gitignore,
        )

        if multiconfig_name_v5.exists():
            print(
                f"| {current_multimanager.naming_complex.expsName} auto-export "
                + 'in "v7" format and remove "v5" format.'
            )
            current_multimanager.write()
            remove_v5_progress = qurry_progressbar(
                old_files.items(),
                bar_format="| {percentage:3.0f}%[{bar}] - remove v5 - {desc} - {elapsed}",
            )
            for k, pathstr in remove_v5_progress:
                if isinstance(pathstr, str):
                    remove_v5_progress.set_description_str(f"{k}")
                    path = Path(pathstr)
                    if path.exists():
                        path.unlink()
                elif isinstance(pathstr, dict):
                    for k2, pathstr2 in pathstr.items():
                        remove_v5_progress.set_description_str(f"{k} - {k2}")
                        path = Path(pathstr2)
                        if path.exists():
                            path.unlink()

        reading_results: list[_E] = experiment_instance.read(  # type: ignore
            save_location=current_multimanager.multicommons.save_location,
            name_or_id=current_multimanager.multicommons.summoner_name,
        )
        for read_exps in reading_results:
            current_multimanager.exps[read_exps.commons.exp_id] = read_exps

        return current_multimanager

    def update_save_location(
        self,
        save_location: Union[Path, str],
        # short_name: str = "",  # TODO: short_name
        without_serial: bool = True,
    ) -> dict[str, Any]:
        """Update the save location of the multi-experiment.

        Args:
            save_location (Union[Path, str]): Location of saving experiment.
            without_serial (bool, optional): Whether without serial number. Defaults to True.

        Returns:
            dict[str, Any]: The dict of multiConfig.
        """
        save_location = Path(save_location)
        self.naming_complex = naming(
            without_serial=without_serial,
            exps_name=self.multicommons.summoner_name,
            # short_name=short_name,
            save_location=save_location,
        )
        self.multicommons = self.multicommons._replace(
            save_location=self.naming_complex.save_location,
            export_location=self.naming_complex.export_location,
        )

        return self.naming_complex._asdict()

    def _write_multiconfig(
        self,
        encoding: str = "utf-8",
        mute: bool = True,
    ) -> dict[str, Any]:
        multiconfig_name = Path(self.multicommons.export_location) / "multi.config.json"
        self.multicommons.files["multi.config"] = str(multiconfig_name)
        self.gitignore.sync("multi.config.json")
        multiconfig = {
            **self.multicommons._asdict(),
            "outfields": self.outfields,
            "files": self.multicommons.files,
        }
        quickJSON(
            content=multiconfig,
            filename=multiconfig_name,
            mode="w+",
            jsonable=True,
            encoding=encoding,
            mute=mute,
        )

        return multiconfig

    def write(
        self,
        save_location: Optional[Union[Path, str]] = None,
        indent: int = 2,
        encoding: str = "utf-8",
        export_transpiled_circuit: bool = False,
        skip_before_and_after: bool = False,
        skip_exps: bool = False,
        skip_quantities: bool = False,
        multiprocess: bool = False,
    ) -> dict[str, Any]:
        """Export the multi-experiment.

        Args:
            save_location (Union[Path, str], optional): Location of saving experiment.
                Defaults to None.
            indent (int, optional): The indent of json file. Defaults to 2.
            encoding (str, optional): The encoding of json file. Defaults to "utf-8".
            export_transpiled_circuit (bool, optional):
                Export the transpiled circuit. Defaults to False.
            skip_before_and_after (bool, optional):
                Skip the beforewards and afterwards. Defaults to False.
            skip_exps (bool, optional):
                Skip the experiments. Defaults to False.
            skip_quantities (bool, optional):
                Skip the quantities container. Defaults to False.
            multiprocess (bool, optional):
                Whether to use multiprocess for exporting. Defaults to False.

        Returns:
            dict[str, Any]: The dict of multiConfig.
        """
        self.gitignore.read(self.multicommons.export_location)
        print("| Export multimanager...")
        if save_location is None:
            save_location = self.multicommons.save_location
        else:
            self.update_save_location(save_location=save_location, without_serial=True)

        self.gitignore.ignore("*.json")
        self.gitignore.sync("qurryinfo.json")
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        if not os.path.exists(self.multicommons.export_location):
            os.makedirs(self.multicommons.export_location)
        self.gitignore.export(self.multicommons.export_location)

        # pylint: disable=protected-access
        exporting_name = {
            **self.afterwards._exporting_name(),
            **self.beforewards._exporting_name(),
        }
        # pylint: enable=protected-access

        export_progress = qurry_progressbar(
            [fname for fname in self.afterwards._fields if fname != "files_taglist"]
            + list(self.beforewards._fields),
            desc="exporting",
            bar_format="qurry-barless",
        )

        # beforewards amd afterwards
        for i, k in enumerate(export_progress):
            if skip_before_and_after or (k in self._unexports):
                export_progress.set_description_str(f"{k} as {exporting_name[k]} - skip")
            elif isinstance(self[k], TagList):
                export_progress.set_description_str(f"{k} as {exporting_name[k]}")
                tmp: TagList = self[k]
                filename = tmp.export(
                    name=None,
                    save_location=self.multicommons.export_location,
                    filetype=self.multicommons.filetype,
                    taglist_name=f"{exporting_name[k]}",
                    open_args={
                        "mode": "w+",
                        "encoding": encoding,
                    },
                    json_dump_args={
                        "indent": indent,
                    },
                )
                self.multicommons.files[exporting_name[k]] = str(filename)
                self.gitignore.sync(f"{exporting_name[k]}.{self.multicommons.filetype}")

            elif isinstance(self[k], (dict, list)):
                export_progress.set_description_str(f"{k} as {exporting_name[k]}")
                filename = Path(self.multicommons.export_location) / f"{exporting_name[k]}.json"
                self.multicommons.files[exporting_name[k]] = str(filename)
                if k not in self._not_sync:
                    self.gitignore.sync(f"{exporting_name[k]}.json")
                quickJSON(
                    content=self[k],
                    filename=filename,
                    mode="w+",
                    jsonable=True,
                    indent=indent,
                    encoding=encoding,
                    mute=True,
                )

            else:
                warnings.warn(f"'{k}' is type '{type(self[k])}' which is not supported to export.")

            if i == len(export_progress) - 1:
                export_progress.set_description_str("exporting done")

        # tagMapQuantity or quantity
        if not skip_quantities:
            self.multicommons.files["quantity"] = self.quantity_container.write(
                save_location=self.multicommons.export_location,
                filetype=self.multicommons.filetype,
                indent=indent,
                encoding=encoding,
            )
            self.gitignore.sync(f"*.quantity.{self.multicommons.filetype}")

        # multiConfig
        multiconfig = self._write_multiconfig(encoding=encoding, mute=True)
        print(f"| Export multi.config.json for {self.summoner_id}")

        self.gitignore.export(self.multicommons.export_location)

        # experiments
        if not skip_exps:
            all_qurryinfo_loc = self.multicommons.export_location / "qurryinfo.json"
            if multiprocess:
                all_export_ids = list(self.beforewards.exps_config.keys())
                first_export = multiprocess_exporter(
                    id_exec=all_export_ids[0],
                    exps_export=self.exps[all_export_ids[0]].export(
                        save_location=self.multicommons.save_location,
                        export_transpiled_circuit=export_transpiled_circuit,
                    ),
                    mode="w+",
                    indent=indent,
                    encoding=encoding,
                    jsonable=True,
                    mute=True,
                    pbar=None,
                )

                exporting_pool = ParallelManager()
                export_list = [
                    (
                        id_exec,
                        self.exps[id_exec].export(
                            save_location=self.multicommons.save_location,
                            export_transpiled_circuit=export_transpiled_circuit,
                        ),
                        "w+",
                        indent,
                        encoding,
                        True,
                        True,
                        None,
                    )
                    for id_exec in all_export_ids[1:]
                ]

                all_qurryinfo_items_since_1 = exporting_pool.process_map(
                    multiprocess_exporter,
                    export_list,
                    bar_format="qurry-barless",
                    desc="Exporting experiments...",
                )
                del export_list
                all_qurryinfo = dict([first_export] + all_qurryinfo_items_since_1)

            else:
                all_qurryinfo = {}
                exps_export_progress = qurry_progressbar(
                    self.beforewards.exps_config,
                    desc="Exporting experiments...",
                    bar_format="qurry-barless",
                )
                for id_exec in exps_export_progress:
                    tmp_id, tmp_qurryinfo_content = single_process_exporter(
                        id_exec=id_exec,
                        exps_export=self.exps[id_exec].export(
                            save_location=self.multicommons.save_location,
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
                self.beforewards.files_taglist[self.exps[id_exec].commons.tags].append(files)
            self.beforewards.files_taglist.export(
                name=None,
                save_location=self.multicommons.export_location,
                taglist_name=f"{exporting_name['files_taglist']}",
                filetype=self.multicommons.filetype,
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
        return multiconfig

    def compress(
        self,
        compress_overwrite: bool = False,
        remain_only_compressed: bool = False,
    ) -> Path:
        """Compress the export_location to tar.xz.

        Args:
            compress_overwrite (bool, optional):
                Reproduce all the compressed files. Defaults to False.
            remain_only_compressed (bool, optional):
                Remove uncompressed files. Defaults to False.

        Returns:
            Path: Path of the compressed file.
        """

        if remain_only_compressed:
            self.multicommons.datetimes.add_serial("uncompressedRemove")
        self._write_multiconfig()

        print(f"| Compress multimanager of '{self.naming_complex.expsName}'...", end="\r")
        loc = self.easycompress(overwrite=compress_overwrite)
        print(f"| Compress multimanager of '{self.naming_complex.expsName}'...done")

        if remain_only_compressed:
            print(
                f"| Remove uncompressed files in '{self.naming_complex.export_location}' ...",
                end="\r",
            )
            shutil.rmtree(self.multicommons.export_location)
            print(f"| Remove uncompressed files in '{self.naming_complex.export_location}' ...done")

        return loc

    def analyze(
        self,
        analysis_name: str = "report",
        no_serialize: bool = False,
        specific_analysis_args: Optional[
            dict[Hashable, Union[dict[str, Any], AnalyzeArgs, bool]]
        ] = None,
        **analysis_args: dict[str, Any],
    ) -> str:
        """Analyze the experiments.

        Args:
            exps_container (ExperimentContainer[_ExpInst]): The container of experiments.
            analysis_name (str, optional): The name of analysis. Defaults to "report".
            no_serialize (bool, optional): Whether serialize the analysis. Defaults to False.
            specific_analysis_args (
                Optional[dict[Hashable, Union[dict[str, Any], bool]]], optional
            ):
                The specific analysis arguments. Defaults to None.
            **analysis_args (dict[str, Any]): The arguments of analysis.

        Returns:
            str: The name of analysis.
        """

        if specific_analysis_args is None:
            specific_analysis_args = {}

        if len(self.afterwards.allCounts) == 0:
            raise ValueError("No counts in multimanagers.")

        idx_tagmap_quantities = len(self.quantity_container)
        name = (
            analysis_name
            if no_serialize
            else f"{analysis_name}." + f"{idx_tagmap_quantities + 1}".rjust(RJUST_LEN, "0")
        )
        self.quantity_container[name] = TagList()

        all_counts_progress = qurry_progressbar(
            self.afterwards.allCounts.keys(),
            bar_format=("| {n_fmt}/{total_fmt} - Analysis: {desc} - {elapsed} < {remaining}"),
        )
        for k in all_counts_progress:

            if k in specific_analysis_args:
                v_args = specific_analysis_args[k]
                if isinstance(v_args, bool):
                    if v_args is False:
                        all_counts_progress.set_description_str(
                            f"Skipped {k} in {self.summoner_id}."
                        )
                        continue
                    report = self.exps[k].analyze(
                        **analysis_args,
                        **({"pbar": all_counts_progress}),
                    )
                else:
                    report = self.exps[k].analyze(
                        **v_args,
                        **({"pbar": all_counts_progress}),
                    )
            else:
                report = self.exps[k].analyze(
                    **analysis_args,
                    **({"pbar": all_counts_progress}),
                )

            main, _tales = report.export(jsonable=False)
            self.quantity_container[name][self.exps[k].commons.tags].append(main)

        self.multicommons.datetimes.add_only(name)

        return name

    def remove_analysis(self, name: str):
        """Removes the analysis.

        Args:
            name (str): The name of the analysis.
        """
        self.quantity_container.remove(name)
        print(f"| Removing analysis: {name}")
        self.multicommons.datetimes.add_only(f"{name}_remove")

    def easycompress(
        self,
        overwrite: bool = False,
    ) -> Path:
        """Compress the export_location to tar.xz.

        Args:
            overwrite (bool, optional): Reproduce all the compressed files. Defaults to False.

        Returns:
            Path: Path of the compressed file.
        """

        self.multicommons.datetimes.add_serial("compressed")
        self._write_multiconfig()

        is_exists = os.path.exists(self.naming_complex.tarLocation)
        if is_exists and overwrite:
            os.remove(self.naming_complex.tarLocation)
            with tarfile.open(self.naming_complex.tarLocation, "x:xz") as tar:
                tar.add(
                    self.naming_complex.export_location,
                    arcname=os.path.basename(self.naming_complex.export_location),
                )

        else:
            with tarfile.open(self.naming_complex.tarLocation, "w:xz") as tar:
                tar.add(
                    self.naming_complex.export_location,
                    arcname=os.path.basename(self.naming_complex.export_location),
                )

        return self.naming_complex.tarLocation

    @classmethod
    def easydecompress(
        cls,
        naming_complex: IOComplex,
    ) -> Path:
        """Decompress the tar.xz file of experiment.

        Args:
            naming_complex (IOComplex): The naming complex of experiment.

        Returns:
            Path: Path of the decompressed file.
        """

        with tarfile.open(naming_complex.tarLocation, "r:xz") as tar:
            tar.extractall(naming_complex.save_location)

        return naming_complex.tarLocation
