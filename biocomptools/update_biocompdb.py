import argparse
import logging
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from sqlmodel import Session
import time
import pandas as pd
from typing import Dict, Optional, Any, List
import shutil
import sqlite3
import xxhash
import sys
import traceback
import json5
from tqdm import tqdm
import os

import biocomp.utils as ut
import biocomptools.toollib.models as md
from biocomptools.toollib.common import config


def parse_args():
    parser = argparse.ArgumentParser(description='BioComp DB Update Script')
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='path to log file (default: <root>/logs/biocomp_db_YYYYMMDD.log)',
    )
    return parser.parse_args()


def setup_logging(log_file_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('biocomp_db')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = RichHandler(
        rich_tracebacks=True, tracebacks_show_locals=False, show_time=True
    )
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    log_file_name = f'biocomp_db_{datetime.now().strftime("%Y%m%d")}.log'
    log_file = (
        Path(log_file_path) if log_file_path else Path(config.paths.root) / 'logs' / log_file_name
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    for name in ['biocomp', 'jax']:
        lib_logger = logging.getLogger(name)
        lib_logger.setLevel(logging.WARNING)
        if not any(isinstance(h, logging.FileHandler) for h in lib_logger.handlers):
            lib_logger.addHandler(file_handler)
        if not any(isinstance(h, RichHandler) for h in lib_logger.handlers):
            lib_logger.addHandler(console_handler)
        lib_logger.propagate = False

    return logger


class BiocompDBUpdater:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.base_dir = Path(config.paths.root).expanduser().resolve()
        self.xp_dir_path = self.base_dir / 'Experiments'
        self.recipe_relative_subpath = 'recipes'

    def _add_item_error(self, item: Any, key: str, detail: str):
        if not hasattr(item, 'errors') or not isinstance(item.errors, dict):
            item.errors = {}
        item.errors.setdefault(key, []).append(detail)

    def _create_experiment_with_error(
        self, xp_cand_dir: Path, error_key: str, log_msg: str
    ) -> md.Experiment:
        self.logger.error(log_msg)
        tb_info = traceback.format_exc()
        try:
            rel_path = xp_cand_dir.relative_to(self.base_dir).as_posix()
        except ValueError:
            self.logger.warning(
                f"experiment dir {xp_cand_dir} not under base {self.base_dir}, using absolute."
            )
            rel_path = xp_cand_dir.as_posix()
        return md.Experiment(
            name=xp_cand_dir.name, path=rel_path, content={}, errors={error_key: [tb_info]}
        )

    def _parse_experiment_file(self, xp_cand_dir: Path) -> Optional[md.Experiment]:
        xp_name = xp_cand_dir.name
        xp_meta_file = xp_cand_dir / 'experiment.json5'

        if not xp_meta_file.exists():
            self.logger.warning(f'no experiment.json5 in {xp_name}, skipping')
            return None
        try:
            with open(xp_meta_file, 'r') as f:
                content = json5.load(f)
            rel_path = xp_cand_dir.relative_to(self.base_dir).as_posix()
            return md.Experiment(name=xp_name, path=rel_path, content=content, errors={})
        except json5.Json5DecodeError as e:
            return self._create_experiment_with_error(
                xp_cand_dir, 'parsing_json5', f"invalid json5 in {xp_meta_file}: {e}"
            )
        except Exception:  # catch all for safety during parsing/init
            return self._create_experiment_with_error(
                xp_cand_dir,
                'parsing_general',
                f"error parsing/creating experiment {xp_name} from {xp_meta_file}",
            )

    def _load_experiments_from_disk(self) -> Dict[str, md.Experiment]:
        experiments: Dict[str, md.Experiment] = {}
        self.logger.info("loading experiments")
        if not self.xp_dir_path.is_dir():
            self.logger.error(f"experiments root {self.xp_dir_path} not found. stopping.")
            return experiments

        exp_cand_dirs = sorted([d for d in self.xp_dir_path.iterdir() if d.is_dir()])

        for xp_cand_dir in tqdm(exp_cand_dirs, desc="loading experiments", unit="dir"):
            xp_name = xp_cand_dir.name
            try:
                xp = self._parse_experiment_file(xp_cand_dir)
                if xp:
                    if xp.name in experiments:
                        self.logger.warning(f"duplicate experiment name {xp.name}, overwriting.")
                    experiments[xp.name] = xp
            except Exception:  # broad catch for safety in loop
                self.logger.error(f"critical unhandled error processing dir {xp_cand_dir}")
                experiments[xp_name] = self._create_experiment_with_error(
                    xp_cand_dir, 'loading_critical', f"critical failure for {xp_name}"
                )

        self.logger.info(f"loaded/processed {len(experiments)} experiments")
        return experiments

    def _validate_datafile_attrs(
        self, attrs: Any, datafile_name: str, xp_name_ctx: str
    ) -> List[str]:
        validation_errors = []
        if not isinstance(attrs, dict):
            validation_errors.append(
                f"datafile {datafile_name} attrs not a dict (type: {type(attrs)})."
            )
            return validation_errors

        spec = {'calibration': ('namehash', dict), 'sample': ('recipe', dict), 'xp': ('name', dict)}
        for key, (subkey, val_type) in spec.items():
            attr_val = attrs.get(key)
            if not isinstance(attr_val, val_type) or subkey not in attr_val:
                validation_errors.append(
                    f"datafile {datafile_name} invalid/missing '{key}' (must be {val_type.__name__} with '{subkey}')."
                )

        if not validation_errors and attrs['xp']['name'] != xp_name_ctx:
            validation_errors.append(
                f"exp name mismatch in {datafile_name}: file has '{attrs['xp']['name']}', context '{xp_name_ctx}'."
            )

        for err in validation_errors:
            self.logger.warning(err)
        return validation_errors

    def _process_single_datafile(
        self, datafile_path: Path, xp: md.Experiment, calibration_map: Dict[str, md.Calibration]
    ):
        try:
            df_content = pd.read_parquet(datafile_path)  # raises if file is not valid parquet
            attr_errors = self._validate_datafile_attrs(
                df_content.attrs, datafile_path.name, xp.name
            )
            if attr_errors:
                self._add_item_error(
                    xp, 'invalid_datafile_attrs', f"{datafile_path.name}: {'; '.join(attr_errors)}"
                )
                return

            cal_attrs = df_content.attrs['calibration']
            namehash = cal_attrs['namehash']
            calib_fullname = f"{xp.name}_{namehash}"
            priority = 1000 if (datafile_path.parent / '.mark_favorite').exists() else 0

            if calib_fullname not in calibration_map:
                calibration_map[calib_fullname] = md.Calibration(
                    fullname=calib_fullname,
                    name=namehash,
                    pipeline=cal_attrs.get('pipeline', {}),
                    data_files=[],
                )

            dfile_rel_path = datafile_path.relative_to(self.base_dir).as_posix()
            dfile_obj = md.DataFile(
                file=dfile_rel_path,
                attrs=df_content.attrs,
                calibration_name=calib_fullname,
                priority=priority,
            )
            calibration_map[calib_fullname].data_files.append(dfile_obj)

        except Exception:
            self.logger.error(f"error processing datafile {datafile_path.name} for xp {xp.name}")
            self._add_item_error(
                xp, 'datafile_processing_error', f"{datafile_path.name}: {traceback.format_exc()}"
            )

    def _link_datafiles_to_recipes(
        self,
        calibration_map: Dict[str, md.Calibration],
        recipe_lookup: Dict[str, md.Recipe],
        xp: md.Experiment,
    ):
        for calib_obj in calibration_map.values():
            if not calib_obj.fullname.startswith(f"{xp.name}_"):
                continue

            for dfile in calib_obj.data_files:
                if not (
                    isinstance(dfile, md.DataFile)
                    and isinstance(getattr(dfile, 'attrs', None), dict)
                ):
                    continue
                try:
                    recipe_name = dfile.attrs.get('sample', {}).get('recipe')
                    if not recipe_name:
                        self.logger.warning(
                            f"missing recipe name in datafile {dfile.file} (xp {xp.name})"
                        )
                        continue

                    target_recipe = recipe_lookup.get(recipe_name)
                    if not target_recipe:
                        self.logger.warning(
                            f"recipe '{recipe_name}' from {dfile.file} not found in {xp.name} recipes."
                        )
                        self._add_item_error(
                            xp, 'recipe_linking_missing', f"recipe '{recipe_name}' for {dfile.file}"
                        )
                        continue

                    dfile.recipe = target_recipe
                    dfile.recipe_name = target_recipe.name
                    if dfile not in target_recipe.data_files:
                        target_recipe.data_files.append(dfile)
                except Exception:
                    self.logger.error(
                        f"error linking recipe for datafile {dfile.file} (xp {xp.name})"
                    )
                    self._add_item_error(
                        xp, 'recipe_linking_error', f"{dfile.file}: {traceback.format_exc()}"
                    )

    def _process_calibrated_data_for_experiment(
        self, xp: md.Experiment, global_calibrations: Dict[str, md.Calibration]
    ):
        recipe_lookup: Dict[str, md.Recipe] = {
            r.content['name']: r
            for r in xp.recipes
            if isinstance(r, md.Recipe) and isinstance(r.content, dict) and 'name' in r.content
        }

        xp_on_disk_path = self.base_dir / xp.path
        for calib_subpath_str in config.calib.paths:
            scan_root_path = xp_on_disk_path / calib_subpath_str
            if not scan_root_path.is_dir():
                continue

            try:
                for datafile_abs_path in scan_root_path.glob('**/*.parquet'):
                    self._process_single_datafile(datafile_abs_path, xp, global_calibrations)
            except Exception:  # broad catch for directory scan issues
                self.logger.error(f"error scanning dir {scan_root_path} for parquet")
                self._add_item_error(
                    xp, 'calibration_scan_error', f"{scan_root_path}: {traceback.format_exc()}"
                )

        self._link_datafiles_to_recipes(global_calibrations, recipe_lookup, xp)

    def _get_db_hash(self, db_file_path: Path) -> str:
        # raises FileNotFoundError if not db_file_path.exists()
        # raises other exceptions on sqlite or xxhash errors
        with sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True) as conn:
            db_dump = '\n'.join(conn.iterdump())
        return xxhash.xxh128(db_dump.encode('utf-8')).hexdigest()

    def _backup_db_if_changed(self) -> bool:
        self.logger.info("checking database for backup")
        db_file = Path(config.db.sqlite.path).expanduser().resolve()
        backup_dir = Path(config.db.sqlite.backup.dir).expanduser().resolve()

        if not db_file.exists():
            self.logger.error(f"db file {db_file} not found. cannot backup.")
            return False

        backup_dir.mkdir(parents=True, exist_ok=True)

        backups = sorted(backup_dir.glob(f'{db_file.stem}_*.sqlite'), key=os.path.getmtime)
        limit = config.db.sqlite.backup.keep_n

        if limit > 0:
            num_to_delete = len(backups) - (limit - 1)
            if num_to_delete > 0:
                for old_bkp in backups[:num_to_delete]:
                    old_bkp.unlink(missing_ok=True)
                    self.logger.info(f"removed old backup: {old_bkp.name}")
                backups = backups[num_to_delete:]

        latest_backup = backups[-1] if backups and limit > 0 else None
        needs_backup = True
        if latest_backup:
            try:
                if self._get_db_hash(latest_backup) == self._get_db_hash(db_file):
                    self.logger.info('no db changes detected since last backup, skipping.')
                    needs_backup = False
            except Exception as e:  # covers FileNotFoundError for hashes too
                self.logger.warning(f"could not compare db hashes ({e}), proceeding with backup.")

        if needs_backup:
            ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            new_backup_file = backup_dir / f'{db_file.stem}_{ts}.sqlite'
            try:
                shutil.copy2(db_file, new_backup_file)
                self.logger.info(f'db backed up to {new_backup_file}')
            except Exception as e:
                self.logger.error(f"error copying db for backup: {e}")
                new_backup_file.unlink(missing_ok=True)  # attempt to clean up partial backup
                return False
        return True

    def _core_processing_loop(
        self,
        experiments: Dict[str, md.Experiment],
        all_calibrations_map: Dict[str, md.Calibration],
        lib: Any,
    ):
        for xp_name, xp_obj in tqdm(
            list(experiments.items()), desc="processing experiments", unit="exp"
        ):
            self.logger.info(f"--- processing experiment: {xp_name} ---")

            try:
                xp_obj.recipes = xp_obj.find_recipes(
                    path_prefix=self.base_dir, recipe_subpath=self.recipe_relative_subpath
                )
            except Exception:
                self.logger.error(f"error finding recipes for {xp_name}")
                self._add_item_error(xp_obj, 'recipe_finding', traceback.format_exc())
                xp_obj.recipes = []

            try:
                self._process_calibrated_data_for_experiment(xp_obj, all_calibrations_map)
            except Exception:
                self.logger.error(f"error processing calibrated data for {xp_name}")
                self._add_item_error(xp_obj, 'calibration_processing', traceback.format_exc())

        all_recipes = [
            r
            for xp in experiments.values()
            for r in getattr(xp, 'recipes', [])
            if isinstance(r, md.Recipe)
        ]
        self.logger.info(f"attempting to build networks for {len(all_recipes)} recipes")
        for recipe in tqdm(all_recipes, desc="building networks", unit="recipe"):
            try:
                recipe.build_networks(
                    lib, inverse='all', use_cache=config.paths.cache.networks, add_to_self=True
                )
                if getattr(recipe, 'errors', {}).get('network_building'):
                    self.logger.warning(f"network building for {recipe.name} had internal errors.")
            except Exception:
                self.logger.error(f"unhandled error building networks for {recipe.name}")
                self._add_item_error(recipe, 'network_building_unhandled', traceback.format_exc())

    def _commit_to_database(self, items_to_merge: List[Any]):
        db_file = Path(config.db.sqlite.path).expanduser().resolve()
        if not db_file.exists():
            self.logger.warning(f"db file {db_file} not found. creating new.")
            md.create_biocompdb_sqlite(db_file, echo=False)

        if not self._backup_db_if_changed():
            self.logger.warning("db backup failed/skipped. proceeding cautiously.")

        engine = md.get_biocompdb_sqlite_engine(db_file, echo=False)
        self.logger.info("--- starting database commit phase ---")

        with Session(engine) as session:
            try:
                for item in tqdm(items_to_merge, desc="merging items to session", unit="item"):
                    try:
                        session.merge(item)
                    except Exception:
                        item_id = getattr(item, 'name', getattr(item, 'fullname', str(item)))
                        self.logger.error(
                            f"error merging {type(item).__name__} '{item_id}'", exc_info=True
                        )
                        self._add_item_error(item, 'db_merge_error', traceback.format_exc())

                session.commit()
                self.logger.info("database transaction committed.")
            except Exception as e:
                self.logger.critical(f"critical error during final db commit: {e}", exc_info=True)
                session.rollback()
                self.logger.info("transaction rolled back.")
                raise

    def run(self):
        run_start_time = time.time()
        self.logger.info(f"--- starting db update run: {datetime.now():%Y-%m-%d %H:%M:%S} ---")

        lib = ut.load_lib()
        experiments = self._load_experiments_from_disk()

        if not experiments:
            self.logger.warning("no experiments loaded. stopping.")
            return

        all_calibrations_map: Dict[str, md.Calibration] = {}
        self._core_processing_loop(experiments, all_calibrations_map, lib)

        items_to_commit = list(experiments.values()) + list(all_calibrations_map.values())
        self._commit_to_database(items_to_commit)

        self.logger.info(f"--- db update run finished in {time.time() - run_start_time:.2f}s ---")


def main():
    cli_args = parse_args()
    try:
        logger = setup_logging(cli_args.log_file)
    except Exception as e:
        print(f"fatal: could not setup logging: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(2)

    try:
        BiocompDBUpdater(logger).run()
        logger.info("Success!")
        sys.exit(0)
    except Exception:
        logger.critical("Terminated with unhandled error in main run", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
