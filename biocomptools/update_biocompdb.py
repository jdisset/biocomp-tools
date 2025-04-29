import argparse
import logging
from pathlib import Path
from datetime import datetime
from rich.logging import RichHandler
from sqlmodel import Session
import time
import pandas as pd
from typing import Dict
from biocomp import utils as ut
from typing import Optional
from tqdm import tqdm
from rich.console import Console
import json5
from biocomptools.toollib.common import config
import biocomptools.toollib.models as md
import shutil
import sqlite3
import xxhash
import sys
import traceback
import os


def parse_args():
    parser = argparse.ArgumentParser(description='BioComp DB Update Script')
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to the log file. If not specified, logs will be written to the default location.',
        default=None,
    )
    return parser.parse_args()


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger('biocomp_db')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True, tracebacks_show_locals=True, show_time=True)
    console_handler.setFormatter(
        logging.Formatter('%(message)s')  # Rich handler adds its own timestamps
    )
    logger.addHandler(console_handler)

    # File handler for persistent logs
    if log_file is None:
        log_file = (
            Path(config.paths.root) / 'logs' / f'biocomp_db_{datetime.now().strftime("%Y%m%d")}.log'
        )
    else:
        log_file = Path(log_file)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    # Set logging levels for other modules

    bch = logging.getLogger('biocomp')
    bch.setLevel(logging.WARNING)
    bch.addHandler(file_handler)
    bch.addHandler(console_handler)
    logging.getLogger('jax').setLevel(logging.WARNING)

    logger.info(f"Logging to file: {log_file}")
    return logger


class BiocompDBUpdater:
    def __init__(self, logger):
        self.logger = logger
        self.console = Console()
        self.BIOCOMP_ROOT = Path(config.paths.root).expanduser().resolve()
        self.DEFAULT_RECIPE_PATH = ['recipes']
        self.base_dir = self.BIOCOMP_ROOT
        self.xp_path = self.base_dir / 'Experiments'
        self.RECIPE_RELATIVE_PATH = 'recipes'
        self.RECIPE_EXT = '.recipe.json5'

    def safe_load_lib(self):
        """Safely load library with error handling"""
        try:
            self.logger.info("Loading library...")
            lib = ut.load_lib()
            self.logger.info("Library loaded successfully")
            return lib
        except Exception as e:
            self.logger.error(f"Failed to load library: {str(e)}")
            self.logger.exception("Library load traceback:")
            raise

    def parse_xp(self, xp_dir: Path) -> Optional[md.Experiment]:
        """Parse experiment with enhanced error handling"""
        experiment = None
        try:
            xpfile = xp_dir / 'experiment.json5'
            if not xpfile.exists():
                self.logger.warning(f'No experiment.json5 file found in {xp_dir.name}. Skipping')
                return None

            filepath = Path(xpfile).expanduser().resolve()
            self.logger.debug(f"Reading experiment file: {filepath}")

            try:
                with open(filepath, 'r') as f:
                    xp_content = json5.load(f)
            except json5.Json5DecodeError as e:
                self.logger.error(f"Invalid JSON5 format in {filepath}: {str(e)}")
                try:
                    experiment = md.Experiment(
                        name=xp_dir.name,
                        path=Path(xp_dir).relative_to(self.base_dir).as_posix(),
                        content={},
                        errors={'parsing': [traceback.format_exc()]},
                    )
                except Exception as inner_e:
                    self.logger.error(
                        f"Could not create basic Experiment object for {xp_dir.name} after JSON error: {inner_e}"
                    )
                return experiment
            except Exception as e:
                self.logger.error(f"Error reading experiment file {filepath}: {str(e)}")
                return None

            experiment = md.Experiment(
                name=xp_dir.name,
                path=Path(xp_dir).relative_to(self.base_dir).as_posix(),
                content=xp_content,
                errors={},
            )
            self.logger.debug(f"Successfully parsed experiment: {experiment.name}")
            return experiment

        except Exception as e:
            self.logger.error(f"Error parsing experiment in {xp_dir}: {str(e)}")
            if experiment:
                if 'parsing' not in experiment.errors:
                    experiment.errors['parsing'] = []
                experiment.errors['parsing'].append(traceback.format_exc())
                return experiment
            try:
                placeholder_xp = md.Experiment(
                    name=xp_dir.name,
                    path=Path(xp_dir).relative_to(self.base_dir).as_posix(),
                    content={},
                    errors={'parsing': [traceback.format_exc()]},
                )
                return placeholder_xp
            except Exception as create_e:
                self.logger.error(
                    f"Could not create placeholder Experiment for {xp_dir.name}: {create_e}"
                )
                return None

    def load_experiments(self) -> Dict[str, md.Experiment]:
        """Load all experiments with progress tracking and error handling"""
        experiments = {}
        self.logger.info("Starting experiment loading process")

        try:
            exp_dirs = sorted([f for f in self.xp_path.iterdir() if f.is_dir()])
            self.logger.info(f"Found {len(exp_dirs)} potential experiment directories")

            for xp_dir in tqdm(exp_dirs, desc="Loading experiments"):
                xp_name = xp_dir.name
                try:
                    xp = self.parse_xp(xp_dir)
                    if xp is not None:
                        if xp.name in experiments:
                            self.logger.warning(
                                f"Duplicate experiment name found: {xp.name}. Overwriting previous entry."
                            )
                        experiments[xp.name] = xp
                except Exception as e:
                    self.logger.error(
                        f"Critical error processing experiment directory {xp_dir}: {str(e)}"
                    )
                    if xp_name not in experiments:
                        try:
                            experiments[xp_name] = md.Experiment(
                                name=xp_name,
                                path=Path(xp_dir).relative_to(self.base_dir).as_posix(),
                                content={},
                                errors={},
                            )
                        except Exception as create_e:
                            self.logger.error(
                                f"Could not create placeholder Experiment for {xp_name} during error handling: {create_e}"
                            )
                            continue
                    if 'processing' not in experiments[xp_name].errors:
                        experiments[xp_name].errors['processing'] = []
                    experiments[xp_name].errors['processing'].append(traceback.format_exc())
                    continue

            self.logger.info(
                f"Finished loading process. Loaded/processed {len(experiments)} experiments"
            )
            return experiments
        except Exception as e:
            self.logger.critical(f"Fatal error during experiment loading phase: {str(e)}")
            self.logger.exception("Experiment loading traceback:")
            raise

    def find_calibrated_data(self, xp: md.Experiment) -> dict[str, md.Calibration]:
        """Find calibrated data with enhanced error handling"""
        self.logger.info(f"Finding calibrated data for experiment: {xp.name}")
        recipe_lookup = {}
        if not hasattr(xp, 'recipes') or xp.recipes is None:
            self.logger.warning(
                f"Experiment {xp.name} has no 'recipes' attribute or it's None. Initializing empty list."
            )
            xp.recipes = []
        if not isinstance(xp.recipes, list):
            self.logger.error(
                f"Experiment {xp.name} 'recipes' attribute is not a list ({type(xp.recipes)}). Resetting to empty list."
            )
            xp.recipes = []

        for recipe in xp.recipes:
            if (
                isinstance(recipe, md.Recipe)
                and isinstance(recipe.content, dict)
                and 'name' in recipe.content
            ):
                recipe_lookup[recipe.content['name']] = recipe
            else:
                self.logger.warning(
                    f"Skipping invalid recipe object in experiment {xp.name}: {recipe}"
                )
                if 'invalid_recipe_obj' not in xp.errors:
                    xp.errors['invalid_recipe_obj'] = []
                xp.errors['invalid_recipe_obj'].append(
                    f"Invalid recipe structure: {str(recipe)[:100]}..."
                )

        calibrations = {}

        for calib_path in config.calib.paths:
            fullpath = Path(self.base_dir) / xp.path / calib_path
            if not fullpath.exists():
                self.logger.debug(
                    f'Calibration path {fullpath} does not exist for {xp.name}, skipping.'
                )
                continue

            self.logger.debug(f"Scanning calibration path: {fullpath}")
            try:
                try:
                    datafile_iterator = fullpath.glob('**/*.parquet')
                except Exception as glob_e:
                    self.logger.error(f"Error scanning directory {fullpath}: {glob_e}")
                    if 'calibration_scan_error' not in xp.errors:
                        xp.errors['calibration_scan_error'] = []
                    xp.errors['calibration_scan_error'].append(
                        f"Path: {fullpath}, Error: {traceback.format_exc()}"
                    )
                    continue

                for datafile in datafile_iterator:
                    try:
                        self.logger.debug(f"Processing datafile: {datafile.name}")
                        priority = 0
                        try:
                            data = pd.read_parquet(datafile)
                        except Exception as read_e:
                            self.logger.error(f"Error reading parquet file {datafile}: {read_e}")
                            if 'datafile_read_error' not in xp.errors:
                                xp.errors['datafile_read_error'] = []
                            xp.errors['datafile_read_error'].append(
                                f"File: {datafile.name}, Error: {traceback.format_exc()}"
                            )
                            continue

                        if not isinstance(data.attrs, dict):
                            self.logger.warning(
                                f"Data file {datafile.name} has invalid attrs type: {type(data.attrs)}. Skipping."
                            )
                            continue
                        if (
                            'calibration' not in data.attrs
                            or not isinstance(data.attrs['calibration'], dict)
                            or 'namehash' not in data.attrs['calibration']
                        ):
                            self.logger.warning(
                                f'Data file {datafile.name} has missing/invalid calibration metadata'
                            )
                            continue
                        if (
                            'sample' not in data.attrs
                            or not isinstance(data.attrs['sample'], dict)
                            or 'recipe' not in data.attrs['sample']
                        ):
                            self.logger.warning(
                                f'Data file {datafile.name} has missing/invalid sample metadata'
                            )
                            continue
                        if (
                            'xp' not in data.attrs
                            or not isinstance(data.attrs['xp'], dict)
                            or 'name' not in data.attrs['xp']
                        ):
                            self.logger.warning(
                                f'Data file {datafile.name} has missing/invalid experiment metadata'
                            )
                            continue

                        # Validate experiment name
                        if data.attrs["xp"]["name"] != xp.name:
                            self.logger.error(
                                f'Experiment name mismatch in {datafile.name}: '
                                f'file says "{data.attrs["xp"]["name"]}", expected "{xp.name}". Skipping.'
                            )
                            if 'xp_mismatch' not in xp.errors:
                                xp.errors['xp_mismatch'] = []
                            xp.errors['xp_mismatch'].append(
                                f"File: {datafile.name}, Expected: {xp.name}, Got: {data.attrs['xp']['name']}"
                            )
                            continue

                        # Check for favorite marking
                        has_favorite = (datafile.parent / '.mark_favorite').exists()
                        if has_favorite:
                            priority = 1000
                            self.logger.debug(f"Marked as favorite: {datafile.name}")

                        namehash = data.attrs['calibration']['namehash']
                        calib_fullname = f"{xp.name}_{namehash}"

                        if namehash not in calibrations:
                            try:
                                calibrations[namehash] = md.Calibration(
                                    fullname=calib_fullname,
                                    name=namehash,
                                    pipeline=data.attrs['calibration'].get('pipeline', {}),
                                    data_files=[],
                                )
                            except Exception as cal_create_e:
                                self.logger.error(
                                    f"Error creating Calibration object for {namehash}: {cal_create_e}"
                                )
                                if 'calibration_create_error' not in xp.errors:
                                    xp.errors['calibration_create_error'] = []
                                xp.errors['calibration_create_error'].append(
                                    f"Namehash: {namehash}, Error: {traceback.format_exc()}"
                                )
                                continue

                        try:
                            dfile = md.DataFile(
                                file=datafile.relative_to(self.base_dir).as_posix(),
                                attrs=data.attrs,
                                calibration_name=calib_fullname,
                                priority=priority,
                            )
                            calibrations[namehash].data_files.append(dfile)
                        except Exception as df_create_e:
                            self.logger.error(
                                f"Error creating DataFile object for {datafile.name}: {df_create_e}"
                            )
                            if 'datafile_create_error' not in xp.errors:
                                xp.errors['datafile_create_error'] = []
                            xp.errors['datafile_create_error'].append(
                                f"File: {datafile.name}, Error: {traceback.format_exc()}"
                            )
                            continue

                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing datafile {datafile.name}: {str(e)}"
                        )
                        if 'datafile_processing_unexpected' not in xp.errors:
                            xp.errors['datafile_processing_unexpected'] = []
                        xp.errors['datafile_processing_unexpected'].append(
                            f"File: {datafile.name}, Error: {traceback.format_exc()}"
                        )
                        continue

                for namehash, calib in calibrations.items():
                    if calib is None or not hasattr(calib, 'data_files') or not calib.data_files:
                        continue

                    for dfile in calib.data_files:
                        if (
                            not isinstance(dfile, md.DataFile)
                            or not hasattr(dfile, 'attrs')
                            or not isinstance(dfile.attrs, dict)
                        ):
                            self.logger.warning(
                                f"Skipping invalid DataFile object during linking: {dfile}"
                            )
                            continue
                        try:
                            # Safe access to attributes
                            sample_info = dfile.attrs.get('sample', {})
                            recipe_name = sample_info.get('recipe')

                            if not recipe_name:
                                self.logger.warning(
                                    f"Missing recipe name in datafile attributes for {dfile.file}"
                                )
                                continue

                            if recipe_name not in recipe_lookup:
                                self.logger.warning(
                                    f"Recipe '{recipe_name}' from datafile {dfile.file} not found in experiment {xp.name}'s loaded recipes."
                                )
                                if 'recipe_linking_missing' not in xp.errors:
                                    xp.errors['recipe_linking_missing'] = []
                                xp.errors['recipe_linking_missing'].append(
                                    f"Recipe '{recipe_name}' not found for datafile {dfile.file}"
                                )
                                continue

                            target_recipe = recipe_lookup[recipe_name]
                            dfile.recipe = target_recipe
                            dfile.recipe_name = target_recipe.name

                            self.logger.debug(
                                f"Linked datafile {dfile.file} to recipe {target_recipe.name}"
                            )

                        except Exception as e:
                            self.logger.error(
                                f"Error linking recipe for datafile {dfile.file}: {str(e)}"
                            )
                            if 'recipe_linking_error' not in xp.errors:
                                xp.errors['recipe_linking_error'] = []
                            xp.errors['recipe_linking_error'].append(
                                f"File: {dfile.file}, Error: {traceback.format_exc()}"
                            )
                            continue

            except Exception as e:
                self.logger.error(f"Error processing calibration path {fullpath}: {str(e)}")
                if 'calibration_path_processing' not in xp.errors:
                    xp.errors['calibration_path_processing'] = []
                xp.errors['calibration_path_processing'].append(
                    f"Path: {fullpath}, Error: {traceback.format_exc()}"
                )
                continue

        found_count = len(calibrations)
        self.logger.info(f"Found {found_count} distinct calibrations for experiment {xp.name}")
        return calibrations

    def get_db_hash(self, db_path: str | Path) -> str:
        """Get database hash with error handling"""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found for hashing: {db_path}")
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # Read-only mode
            db_full_dump = '\n'.join(conn.iterdump())
            conn.close()
            return xxhash.xxh128(db_full_dump.encode('utf-8')).hexdigest()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error calculating database hash for {db_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error calculating database hash for {db_path}: {str(e)}")
            raise

    def backup_db_if_changed(
        self,
        db_path=config.db.sqlite.path,
        db_backup_dir=config.db.sqlite.backup.dir,
        backup_limit=config.db.sqlite.backup.keep_n,
    ):
        """Backup database with enhanced error handling and backup rotation."""
        try:
            self.logger.info("Starting database backup process")
            db_path = Path(db_path).expanduser().resolve()
            db_backup_dir = Path(db_backup_dir).expanduser().resolve()

            if not db_path.exists():
                self.logger.error(f"Database file does not exist: {db_path}. Cannot backup.")
                return False  # Indicate failure

            db_backup_dir.mkdir(parents=True, exist_ok=True)

            # rotation: clean up old backups first
            try:
                existing_backups = sorted(
                    db_backup_dir.glob(f'{db_path.stem}_*.sqlite'), key=os.path.getmtime
                )
                if backup_limit > 0 and len(existing_backups) >= backup_limit:
                    num_to_delete = len(existing_backups) - backup_limit + 1
                    for old_backup in existing_backups[:num_to_delete]:
                        try:
                            old_backup.unlink()
                            self.logger.info(f"Removed old backup: {old_backup.name}")
                        except OSError as delete_err:
                            self.logger.error(
                                f"Error removing old backup {old_backup.name}: {delete_err}"
                            )
            except Exception as rotation_err:
                self.logger.error(f"Error during backup rotation: {rotation_err}")

            latest_backup = existing_backups[-1] if existing_backups else None
            needs_backup = True

            if latest_backup is not None:
                try:
                    latest_backup_hash = self.get_db_hash(latest_backup)
                    current_hash = self.get_db_hash(db_path)
                    if latest_backup_hash == current_hash:
                        self.logger.info(
                            'No changes detected in database since last backup, skipping backup.'
                        )
                        needs_backup = False
                except Exception as e:
                    self.logger.warning(
                        f"Could not compare database hashes: {str(e)}. Proceeding with backup."
                    )
                    pass

            if needs_backup:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                new_db_backup_path = db_backup_dir / f'{db_path.stem}_{timestamp}.sqlite'
                try:
                    self.logger.info(f'Backing up database to {new_db_backup_path}')
                    shutil.copy2(db_path, new_db_backup_path)  # use copy2 to preserve metadata
                    self.logger.info("Database backup completed successfully")
                    return True  # Indicate success
                except Exception as copy_err:
                    self.logger.error(f"Error copying database file during backup: {copy_err}")
                    if new_db_backup_path.exists():
                        try:
                            new_db_backup_path.unlink()
                        except OSError:
                            pass
                    return False
            else:
                return True

        except Exception as e:
            self.logger.error(f"Unexpected error during database backup process: {str(e)}")
            self.logger.exception("Backup traceback:")
            return False

    def run(self):
        """Main execution with comprehensive error handling"""
        start_time = time.time()
        self.logger.info(
            f"--- Starting Database Update Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )

        lib = None
        experiments = {}
        engine = None

        try:
            lib = self.safe_load_lib()

            # load Experiments
            experiments = self.load_experiments()
            if not experiments:
                self.logger.warning("No experiments were loaded. Stopping.")
                return  # Nothing to do if no experiments

            # process recipes & calibrated data for each Experiment
            total_recipes_found = 0
            processed_experiments = {}
            for xp_name, xp in experiments.items():
                self.logger.info(f"--- Processing Experiment: {xp_name} ---")
                experiment_failed = False
                if not hasattr(xp, 'errors'):
                    xp.errors = {}

                try:
                    xp.recipes = xp.find_recipes(
                        path_prefix=self.base_dir, recipe_subpath=self.RECIPE_RELATIVE_PATH
                    )
                    total_recipes_found += len(xp.recipes)
                    self.logger.info(f"Found {len(xp.recipes)} recipes for {xp_name}")
                except Exception as e_find:
                    self.logger.error(
                        f"Error finding recipes for experiment {xp_name}: {str(e_find)}"
                    )
                    if 'recipe_finding' not in xp.errors:
                        xp.errors['recipe_finding'] = []
                    xp.errors['recipe_finding'].append(traceback.format_exc())
                    xp.recipes = []
                    experiment_failed = True

                # find calibrated data (runs even if recipe finding failed, might log errors)
                try:
                    self.find_calibrated_data(xp)
                except Exception as e_calib:
                    self.logger.error(
                        f"Error finding calibrated data for experiment {xp_name}: {str(e_calib)}"
                    )
                    if 'calibration_finding' not in xp.errors:
                        xp.errors['calibration_finding'] = []
                    xp.errors['calibration_finding'].append(traceback.format_exc())
                    experiment_failed = True

                if not experiment_failed:
                    processed_experiments[xp_name] = xp
                else:
                    self.logger.warning(
                        f"Experiment {xp_name} encountered errors during initial processing, may be incomplete."
                    )

            self.logger.info(
                f'Found {total_recipes_found} recipes total across all processed experiments'
            )

            # build Networks (only for successfully processed experiments)
            recipes_to_build_count = sum(
                len(xp.recipes)
                for xp in processed_experiments.values()
                if hasattr(xp, 'recipes') and xp.recipes
            )
            self.logger.info(f"Attempting to build networks for {recipes_to_build_count} recipes.")
            progress = tqdm(total=recipes_to_build_count, desc="Building networks")
            for xp_name, xp in processed_experiments.items():
                if not hasattr(xp, 'recipes') or not xp.recipes:
                    continue  # skip if no recipes were loaded for this exp

                for recipe in xp.recipes:
                    # Ensure recipe and errors exist
                    if not isinstance(recipe, md.Recipe):
                        continue
                    if not hasattr(recipe, 'errors'):
                        recipe.errors = {}

                    try:
                        if 'network_building' in recipe.errors:
                            del recipe.errors['network_building']

                        networks = recipe.build_networks(
                            lib,
                            inverse='all',
                            use_cache=config.paths.cache.networks,
                            add_to_self=True,  # links networks to the recipe
                        )
                        # check if build_networks added errors internally
                        if recipe.errors.get('network_building'):
                            self.logger.warning(
                                f"Network building for recipe {recipe.name} completed with internal errors."
                            )
                        else:
                            self.logger.debug(
                                f"Successfully built {len(networks)} networks for recipe {recipe.name}"
                            )

                    except Exception as e:
                        self.logger.error(
                            f"Unhandled error during build_networks call for recipe {recipe.name}: {str(e)}"
                        )
                        if 'network_building' not in recipe.errors:
                            recipe.errors['network_building'] = []
                        recipe.errors['network_building'].append(
                            f"Unhandled Exception: {traceback.format_exc()}"
                        )
                    finally:
                        progress.update(1)
            progress.close()

            db_path = Path(config.db.sqlite.path).expanduser().resolve()
            if not db_path.exists():
                self.logger.warning(f"Database file {db_path} not found. Creating new database.")
                md.create_biocompdb_sqlite(db_path, echo=False)  # Don't echo schema creation

            # backup before making changes
            backup_success = self.backup_db_if_changed()
            if not backup_success:
                self.logger.warning(
                    "Backup failed or was skipped. Proceeding with DB update cautiously."
                )

            engine = md.get_biocompdb_sqlite_engine(db_path, echo=False)

            self.logger.info("--- Starting Database Commit Phase ---")
            objects_to_commit_count = len(experiments)
            commit_progress = tqdm(total=objects_to_commit_count, desc="Adding/Merging to session")
            with Session(engine) as session:
                try:
                    for xp_name, xp in experiments.items():
                        try:
                            session.merge(xp)
                            commit_progress.update(1)

                        except Exception as e:
                            self.logger.error(
                                f"Error merging experiment {xp_name} into session: {str(e)}"
                            )
                            self.logger.exception(f"Traceback for merging {xp_name}:")
                            # add merge error to the object itself if possible
                            if hasattr(xp, 'errors') and isinstance(xp.errors, dict):
                                if 'db_merge_error' not in xp.errors:
                                    xp.errors['db_merge_error'] = []
                                xp.errors['db_merge_error'].append(traceback.format_exc())
                            commit_progress.update(1)
                            continue  # skip this xp

                    commit_progress.close()
                    self.logger.info("Committing transaction to database...")
                    session.commit()
                    self.logger.info("Database transaction committed successfully.")

                except Exception as e:
                    commit_progress.close()
                    self.logger.critical(f"CRITICAL ERROR during final database commit: {str(e)}")
                    self.logger.exception("Commit traceback:")
                    self.logger.info("Rolling back transaction...")
                    session.rollback()
                    self.logger.info("Transaction rolled back.")
                    raise  # Re-raise after rollback

            total_time = time.time() - start_time
            self.logger.info(f"--- Database Update Run Finished ---")
            self.logger.info(f"Total execution time: {total_time:.2f} seconds")

        except Exception as e:
            self.logger.critical(f"Fatal error in main execution run: {str(e)}", exc_info=True)
            raise


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)

    try:
        app = BiocompDBUpdater(logger)
        app.run()
        logger.info("Application finished successfully.")
        sys.exit(0)
    except Exception as e:
        logger.critical(
            f"Application terminated with unhandled error in main: {str(e)}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
