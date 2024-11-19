import dracon as dr
from sqlmodel import Session
import time
import pandas as pd
from typing import List, Tuple, TypeVar, Dict
from biocomp import utils as ut
import biocomp as bc
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import numpy as np
from rich import print as rprint
from rich.console import Console
from pydantic import BaseModel
import logging
import json5
from biocomptools.toollib.common import config
import biocomptools.toollib.models as md
import shutil
import sqlite3
import xxhash
from datetime import datetime


# Configure logging
def setup_logging() -> logging.Logger:
    logger = logging.getLogger('biocomp_db')
    logger.setLevel(logging.INFO)

    # Console handler with rich formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

    # File handler for persistent logs
    log_file = (
        Path(config.paths.root) / 'logs' / f'biocomp_db_{datetime.now().strftime("%Y%m%d")}.log'
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    return logger


logger = setup_logging()

# Set logging levels for other modules
logging.getLogger('biocomp').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)

console = Console()
BIOCOMP_ROOT = Path(config.paths.root).expanduser().resolve()
DEFAULT_RECIPE_PATH = ['recipes']
base_dir = BIOCOMP_ROOT
xp_path = base_dir / 'Experiments'
RECIPE_RELATIVE_PATH = 'recipes'
RECIPE_EXT = '.recipe.json5'


def safe_load_lib():
    """Safely load library with error handling"""
    try:
        logger.info("Loading library...")
        lib = ut.load_lib()
        logger.info("Library loaded successfully")
        return lib
    except Exception as e:
        logger.error(f"Failed to load library: {str(e)}")
        raise


lib = safe_load_lib()


def parse_xp(xp_dir: Path) -> Optional[md.Experiment]:
    """Parse experiment with enhanced error handling"""
    try:
        xpfile = xp_dir / 'experiment.json5'
        if not xpfile.exists():
            logger.warning(f'No experiment.json5 file found in {xp_dir.name}. Skipping')
            return None

        filepath = Path(xpfile).expanduser().resolve()
        logger.debug(f"Reading experiment file: {filepath}")

        try:
            with open(filepath, 'r') as f:
                xp = json5.load(f)
        except json5.Json5DecodeError as e:
            logger.error(f"Invalid JSON5 format in {filepath}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error reading experiment file {filepath}: {str(e)}")
            return None

        experiment = md.Experiment(
            name=xp_dir.name,
            path=Path(xp_dir).relative_to(base_dir).as_posix(),
            content=xp,
        )
        logger.debug(f"Successfully parsed experiment: {experiment.name}")
        return experiment

    except Exception as e:
        logger.error(f"Error parsing experiment in {xp_dir}: {str(e)}")
        return None


def load_experiments() -> Dict[str, md.Experiment]:
    """Load all experiments with progress tracking and error handling"""
    experiments = {}
    logger.info("Starting experiment loading process")

    try:
        exp_dirs = sorted([f for f in xp_path.iterdir() if f.is_dir()])
        logger.info(f"Found {len(exp_dirs)} potential experiment directories")

        for xp_dir in tqdm(exp_dirs, desc="Loading experiments"):
            try:
                xp = parse_xp(xp_dir)
                if xp is not None:
                    if xp.name in experiments:
                        logger.error(f"Duplicate experiment name found: {xp.name}")
                        continue
                    experiments[xp.name] = xp
            except Exception as e:
                logger.error(f"Error processing experiment directory {xp_dir}: {str(e)}")
                continue

        logger.info(f"Successfully loaded {len(experiments)} experiments")
        return experiments
    except Exception as e:
        logger.error(f"Fatal error loading experiments: {str(e)}")
        raise


def find_calibrated_data(xp: md.Experiment) -> dict[str, md.Calibration]:
    """Find calibrated data with enhanced error handling"""
    logger.info(f"Finding calibrated data for experiment: {xp.name}")
    recipe_lookup = {recipe.content['name']: recipe for recipe in xp.recipes}
    calibrations = {}

    for calib_path in config.calib.paths:
        fullpath = Path(base_dir) / xp.path / calib_path
        if not fullpath.exists():
            logger.warning(f'Calibration path {fullpath} does not exist')
            continue

        try:
            for datafile in fullpath.glob('**/*.parquet'):
                try:
                    logger.debug(f"Processing datafile: {datafile}")
                    priority = 0
                    data = pd.read_parquet(datafile)

                    # Validate file metadata
                    if 'calibration' not in data.attrs:
                        logger.warning(f'Data file {datafile} has no calibration metadata')
                        continue
                    if 'sample' not in data.attrs:
                        logger.warning(f'Data file {datafile} has no sample metadata')
                        continue

                    # Validate experiment name
                    if data.attrs["xp"]["name"] != xp.name:
                        logger.error(
                            f'Experiment name mismatch in {datafile}: '
                            f'{data.attrs["xp"]["name"]} != {xp.name}'
                        )
                        continue

                    # Check for favorite marking
                    has_favorite = (datafile.parent / '.mark_favorite').exists()
                    if has_favorite:
                        priority = 1000
                        logger.debug(f"Marked as favorite: {datafile}")

                    namehash = data.attrs['calibration']['namehash']
                    calib_name = f"{xp.name}_{namehash}"

                    if namehash not in calibrations:
                        calibrations[namehash] = md.Calibration(
                            name=calib_name,
                            pipeline=data.attrs['calibration']['pipeline'],
                        )
                        calibrations[namehash].data_files = []

                    dfile = md.DataFile(
                        file=datafile.relative_to(base_dir).as_posix(),
                        attrs=data.attrs,
                        calibration_name=calib_name,
                        priority=priority,
                    )

                    calibrations[namehash].data_files.append(dfile)

                except Exception as e:
                    logger.error(f"Error processing datafile {datafile}: {str(e)}")
                    continue

            # Link recipes to data files
            for calib in calibrations.values():
                for dfile in calib.data_files:
                    try:
                        recipe_name = dfile.attrs['sample']['recipe']
                        if recipe_name not in recipe_lookup:
                            logger.error(f"Recipe {recipe_name} not found in lookup")
                            continue

                        dfile.recipe = recipe_lookup[recipe_name]
                        if id(dfile.recipe.data_files[-1]) != id(dfile):
                            logger.error("Recipe data_files not linked correctly")

                    except Exception as e:
                        logger.error(f"Error linking recipe for {dfile.file}: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"Error processing calibration path {calib_path}: {str(e)}")
            continue

    logger.info(f"Found {len(calibrations)} calibrations for experiment {xp.name}")
    return calibrations


def get_db_hash(db_path: str | Path) -> str:
    """Get database hash with error handling"""
    try:
        db_path = Path(db_path)
        conn = sqlite3.connect(db_path)
        db_full_dump = '\n'.join(conn.iterdump())
        conn.close()
        return xxhash.xxh128(db_full_dump).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating database hash: {str(e)}")
        raise


def backup_db_if_changed(db_path=config.db.sqlite.path, db_backup_dir=config.db.sqlite.backup.dir):
    """Backup database with enhanced error handling"""
    try:
        logger.info("Starting database backup process")
        db_path = Path(db_path).expanduser().resolve()
        db_backup_dir = Path(db_backup_dir).expanduser().resolve()

        if not db_path.exists():
            logger.error(f"Database file does not exist: {db_path}")
            return

        db_backup_dir.mkdir(parents=True, exist_ok=True)
        existing_backups = sorted(db_backup_dir.glob(f'{db_path.stem}_*.sqlite'))
        latest_backup = None if not existing_backups else existing_backups[-1]

        new_db_backup_path = (
            db_backup_dir / f'{db_path.stem}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}.sqlite'
        )

        if latest_backup is not None:
            try:
                latest_backup_hash = get_db_hash(latest_backup)
                current_hash = get_db_hash(db_path)
                if latest_backup_hash == current_hash:
                    logger.info('No changes in database, skipping backup')
                    return
            except Exception as e:
                logger.error(f"Error comparing database hashes: {str(e)}")
                # Continue with backup anyway
                pass

        logger.info(f'Backing up database to {new_db_backup_path}')
        shutil.copy(db_path, new_db_backup_path)
        logger.info("Database backup completed successfully")

    except Exception as e:
        logger.error(f"Error during database backup: {str(e)}")
        raise


def main():
    """Main execution with comprehensive error handling"""
    start_time = time.time()
    logger.info("Starting database update process")

    try:
        # Load experiments
        experiments = load_experiments()

        # Process recipes and build networks
        total_recipes = 0
        for xp in experiments.values():
            try:
                logger.info(f"Processing recipes for experiment: {xp.name}")
                xp.recipes = xp.find_recipes(
                    path_prefix=base_dir, recipe_subpath=RECIPE_RELATIVE_PATH
                )
                find_calibrated_data(xp)
                total_recipes += len(xp.recipes)
            except Exception as e:
                logger.error(f"Error processing recipes for experiment {xp.name}: {str(e)}")
                continue

        logger.info(f'Found {total_recipes} recipes total')

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
                    logger.error(
                        f"Error building networks for recipe {recipe.content['name']}: {str(e)}"
                    )
                    progress.update(1)
                    continue
        progress.close()

        # Database operations
        if not Path(config.db.sqlite.path).exists():
            logger.warning(
                f"Database file {config.db.sqlite.path} not found. Creating new database"
            )
            md.create_biocompdb_sqlite(config.db.sqlite.path, echo=True)

        backup_db_if_changed()
        engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, True)

        with Session(engine) as session:
            try:
                for xp in experiments.values():
                    try:
                        session.add(xp)
                    except Exception as e:
                        logger.error(f"Error adding experiment {xp.name} to session: {str(e)}")
                        continue

                logger.info("Committing changes to database")
                session.commit()
                logger.info("Database update completed successfully")

            except Exception as e:
                logger.error(f"Error during database commit: {str(e)}")
                logger.info("Rolling back transaction")
                session.rollback()
                raise

        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application terminated with error: {str(e)}")
        raise
