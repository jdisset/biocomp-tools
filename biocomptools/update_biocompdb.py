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
            raise

    def parse_xp(self, xp_dir: Path) -> Optional[md.Experiment]:
        """Parse experiment with enhanced error handling"""
        try:
            xpfile = xp_dir / 'experiment.json5'
            if not xpfile.exists():
                self.logger.warning(f'No experiment.json5 file found in {xp_dir.name}. Skipping')
                return None

            filepath = Path(xpfile).expanduser().resolve()
            self.logger.debug(f"Reading experiment file: {filepath}")

            try:
                with open(filepath, 'r') as f:
                    xp = json5.load(f)
            except json5.Json5DecodeError as e:
                self.logger.error(f"Invalid JSON5 format in {filepath}: {str(e)}")
                return None
            except Exception as e:
                self.logger.error(f"Error reading experiment file {filepath}: {str(e)}")
                return None

            experiment = md.Experiment(
                name=xp_dir.name,
                path=Path(xp_dir).relative_to(self.base_dir).as_posix(),
                content=xp,
            )
            self.logger.debug(f"Successfully parsed experiment: {experiment.name}")
            return experiment

        except Exception as e:
            self.logger.error(f"Error parsing experiment in {xp_dir}: {str(e)}")
            return None

    def load_experiments(self) -> Dict[str, md.Experiment]:
        """Load all experiments with progress tracking and error handling"""
        experiments = {}
        self.logger.info("Starting experiment loading process")

        try:
            exp_dirs = sorted([f for f in self.xp_path.iterdir() if f.is_dir()])
            self.logger.info(f"Found {len(exp_dirs)} potential experiment directories")

            for xp_dir in tqdm(exp_dirs, desc="Loading experiments"):
                try:
                    xp = self.parse_xp(xp_dir)
                    if xp is not None:
                        if xp.name in experiments:
                            self.logger.error(f"Duplicate experiment name found: {xp.name}")
                            continue
                        experiments[xp.name] = xp
                except Exception as e:
                    self.logger.error(f"Error processing experiment directory {xp_dir}: {str(e)}")
                    continue

            self.logger.info(f"Successfully loaded {len(experiments)} experiments")
            return experiments
        except Exception as e:
            self.logger.error(f"Fatal error loading experiments: {str(e)}")
            raise

    def find_calibrated_data(self, xp: md.Experiment) -> dict[str, md.Calibration]:
        """Find calibrated data with enhanced error handling"""
        self.logger.info(f"Finding calibrated data for experiment: {xp.name}")
        recipe_lookup = {recipe.content['name']: recipe for recipe in xp.recipes}
        calibrations = {}

        for calib_path in config.calib.paths:
            fullpath = Path(self.base_dir) / xp.path / calib_path
            if not fullpath.exists():
                self.logger.warning(f'Calibration path {fullpath} does not exist')
                continue

            try:
                for datafile in fullpath.glob('**/*.parquet'):
                    try:
                        self.logger.debug(f"Processing datafile: {datafile}")
                        priority = 0
                        data = pd.read_parquet(datafile)

                        # Validate file metadata
                        if 'calibration' not in data.attrs:
                            self.logger.warning(f'Data file {datafile} has no calibration metadata')
                            continue
                        if 'sample' not in data.attrs:
                            self.logger.warning(f'Data file {datafile} has no sample metadata')
                            continue

                        # Validate experiment name
                        if data.attrs["xp"]["name"] != xp.name:
                            self.logger.error(
                                f'Experiment name mismatch in {datafile}: '
                                f'{data.attrs["xp"]["name"]} != {xp.name}'
                            )
                            continue

                        # Check for favorite marking
                        has_favorite = (datafile.parent / '.mark_favorite').exists()
                        if has_favorite:
                            priority = 1000
                            self.logger.debug(f"Marked as favorite: {datafile}")

                        namehash = data.attrs['calibration']['namehash']
                        calib_name = f"{xp.name}_{namehash}"

                        if namehash not in calibrations:
                            calibrations[namehash] = md.Calibration(
                                name=calib_name,
                                pipeline=data.attrs['calibration']['pipeline'],
                            )
                            calibrations[namehash].data_files = []

                        dfile = md.DataFile(
                            file=datafile.relative_to(self.base_dir).as_posix(),
                            attrs=data.attrs,
                            calibration_name=calib_name,
                            priority=priority,
                        )

                        calibrations[namehash].data_files.append(dfile)

                    except Exception as e:
                        self.logger.error(f"Error processing datafile {datafile}: {str(e)}")
                        continue

                # Link recipes to data files
                for calib in calibrations.values():
                    for dfile in calib.data_files:
                        try:
                            recipe_name = dfile.attrs['sample']['recipe']
                            if recipe_name not in recipe_lookup:
                                self.logger.error(f"Recipe {recipe_name} not found in lookup")
                                continue

                            dfile.recipe = recipe_lookup[recipe_name]
                            if id(dfile.recipe.data_files[-1]) != id(dfile):
                                self.logger.error("Recipe data_files not linked correctly")

                        except Exception as e:
                            self.logger.error(f"Error linking recipe for {dfile.file}: {str(e)}")
                            continue

            except Exception as e:
                self.logger.error(f"Error processing calibration path {calib_path}: {str(e)}")
                continue

        self.logger.info(f"Found {len(calibrations)} calibrations for experiment {xp.name}")
        return calibrations

    def get_db_hash(self, db_path: str | Path) -> str:
        """Get database hash with error handling"""
        try:
            db_path = Path(db_path)
            conn = sqlite3.connect(db_path)
            db_full_dump = '\n'.join(conn.iterdump())
            conn.close()
            return xxhash.xxh128(db_full_dump).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating database hash: {str(e)}")
            raise

    def backup_db_if_changed(
        self, db_path=config.db.sqlite.path, db_backup_dir=config.db.sqlite.backup.dir
    ):
        """Backup database with enhanced error handling"""
        try:
            self.logger.info("Starting database backup process")
            db_path = Path(db_path).expanduser().resolve()
            db_backup_dir = Path(db_backup_dir).expanduser().resolve()

            if not db_path.exists():
                self.logger.error(f"Database file does not exist: {db_path}")
                return

            db_backup_dir.mkdir(parents=True, exist_ok=True)
            existing_backups = sorted(db_backup_dir.glob(f'{db_path.stem}_*.sqlite'))
            latest_backup = None if not existing_backups else existing_backups[-1]

            new_db_backup_path = (
                db_backup_dir / f'{db_path.stem}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}.sqlite'
            )

            if latest_backup is not None:
                try:
                    latest_backup_hash = self.get_db_hash(latest_backup)
                    current_hash = self.get_db_hash(db_path)
                    if latest_backup_hash == current_hash:
                        self.logger.info('No changes in database, skipping backup')
                        return
                except Exception as e:
                    self.logger.error(f"Error comparing database hashes: {str(e)}")
                    # Continue with backup anyway
                    pass

            self.logger.info(f'Backing up database to {new_db_backup_path}')
            shutil.copy(db_path, new_db_backup_path)
            self.logger.info("Database backup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during database backup: {str(e)}")
            raise

    def run(self):
        """Main execution with comprehensive error handling"""
        start_time = time.time()
        self.logger.info("Starting database update process")

        try:
            # Load library
            lib = self.safe_load_lib()

            # Load experiments
            experiments = self.load_experiments()

            # Process recipes and build networks
            total_recipes = 0
            for xp in experiments.values():
                try:
                    self.logger.info(f"Processing recipes for experiment: {xp.name}")
                    xp.recipes = xp.find_recipes(
                        path_prefix=self.base_dir, recipe_subpath=self.RECIPE_RELATIVE_PATH
                    )
                    self.find_calibrated_data(xp)
                    total_recipes += len(xp.recipes)
                except Exception as e:
                    self.logger.error(
                        f"Error processing recipes for experiment {xp.name}: {str(e)}"
                    )
                    continue

            self.logger.info(f'Found {total_recipes} recipes total')

            # Build networks
            progress = tqdm(total=total_recipes)
            for xp in experiments.values():
                progress.set_description(f'{xp.name}')
                for recipe in xp.recipes:
                    try:
                        networks = recipe.build_networks(
                            lib,
                            inverse='all',
                            use_cache=config.paths.cache.networks,
                            add_to_self=True,
                        )
                        progress.update(1)
                    except Exception as e:
                        self.logger.error(
                            f"Error building networks for recipe {recipe.content['name']}: {str(e)}"
                        )
                        progress.update(1)
                        continue
            progress.close()

            # Database operations
            if not Path(config.db.sqlite.path).exists():
                self.logger.warning(
                    f"Database file {config.db.sqlite.path} not found. Creating new database"
                )
                md.create_biocompdb_sqlite(config.db.sqlite.path, echo=True)

            self.backup_db_if_changed()
            engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, True)

            with Session(engine) as session:
                try:
                    for xp in experiments.values():
                        try:
                            session.add(xp)
                        except Exception as e:
                            self.logger.error(
                                f"Error adding experiment {xp.name} to session: {str(e)}"
                            )
                            continue

                    self.logger.info("Committing changes to database")
                    session.commit()
                    self.logger.info("Database update completed successfully")

                except Exception as e:
                    self.logger.error(f"Error during database commit: {str(e)}")
                    self.logger.info("Rolling back transaction")
                    session.rollback()
                    raise

            total_time = time.time() - start_time
            self.logger.info(f"Total execution time: {total_time:.2f} seconds")

        except Exception as e:
            self.logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
            raise


def main():
    # Parse arguments first
    args = parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)

    # Create and run the application
    try:
        app = BiocompDBUpdater(logger)
        app.run()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # We don't use logger here because it might not be initialized if setup_logging failed
        print(f"Application terminated with error: {str(e)}", file=sys.stderr)
        raise
