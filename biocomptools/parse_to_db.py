## {{{                           --     TODO     --
"""
---
# TODO:
- [x] Figure out question: should Network even exist? Should recipe be tied to xp instead? Maybe Sample?
- [x] make Recipe a SQLModel
> - [x] it contains the actual recipe in the content field (json) + hash
> - [x] it has a unique name of {xp}_{recipe}
>    that means there can be content duplicates but that's fine, worst case I can merge by hash later
> - [x] it can build networks (and returns a md.Network with its _network attribute set)
>    - make it so that it builds all the networks,
>      preferably using the content rather than the file (much more portable)
> - [x] properly link networks and recipes (a network is linked to a single recipe)
> - [x] properly link recipes and data files (a data file is linked to a single recipe)

- [x] finish the find_calibrated_data function (parse all calib data)
---
> [!NOTE]
> Ultimately, we should merge the main db with the parts db, and link recipes to parts

---
"""

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                          --     imports     --

import dracon as dr
from sqlmodel import Session
import time
import pandas as pd
from typing import List, Tuple, TypeVar
from biocomp import utils as ut
import biocomp as bc
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import numpy as np

# pretty print from rich
from rich import print as rprint
from pathlib import Path
from rich import print as rprint
from pydantic import BaseModel

import logging
from biocomptools.toollib.common import config
import biocomptools.toollib.models as md
from rich.console import Console

logger = logging.getLogger('build_xp_table')
logger.setLevel(logging.DEBUG)
logging.getLogger('biocomp').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)

console = Console()
BIOCOMP_ROOT = Path(config.paths.root).expanduser().resolve()
DEFAULT_RECIPE_PATH = ['recipes']
base_dir = BIOCOMP_ROOT
xp_path = base_dir / 'Experiments'
RECIPE_RELATIVE_PATH = 'recipes'
lib = ut.load_lib()
RECIPE_EXT = '.recipe.json5'

##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     list all xps in experiment folder    --


def parse_xp(xp_dir: Path) -> Optional[md.Experiment]:
    import json5

    xpfile = xp_dir / 'experiment.json5'
    if not xpfile.exists():
        logger.warning(f'No experiment.json5 file found in {xp_dir.name}. Skipping')
        return None

    filepath = Path(xpfile).expanduser().resolve()
    with open(filepath, 'r') as f:
        xp = json5.load(f)

    return md.Experiment(
        name=xp_dir.name,
        path=Path(xp_dir).relative_to(base_dir).as_posix(),
        content=xp,
    )


experiments = {}

for xp_dir in tqdm(sorted([f for f in xp_path.iterdir() if f.is_dir()])):
    xp = parse_xp(xp_dir)
    if xp is not None:
        assert xp.name not in experiments
        experiments[xp.name] = xp

logger.info(f'found {len(experiments)} experiments')


##────────────────────────────────────────────────────────────────────────────}}}

### {{{  --     find data    --


def find_calibrated_data(xp: md.Experiment) -> dict[str, md.Calibration]:
    recipe_lookup = {recipe.content['name']: recipe for recipe in xp.recipes}
    calibrations = {}
    for calib_path in config.calib.paths:
        fullpath = Path(base_dir) / xp.path / calib_path
        if not fullpath.exists():
            logger.warning(f'Calibration path {fullpath} does not exist')
            continue

        # find all data files
        for datafile in fullpath.glob('**/*.parquet'):
            priority = 0
            data = pd.read_parquet(datafile)
            # check if there is a .mark_favorite file in the same folder
            has_favorite = (datafile.parent / '.mark_favorite').exists()
            if has_favorite:
                priority = 1000

            if 'calibration' not in data.attrs:
                logger.warning(f'Data file {datafile} has no calibration metadata')
                continue

            if 'sample' not in data.attrs:
                logger.warning(f'Data file {datafile} has no sample metadata')
                continue

            assert (
                data.attrs["xp"]["name"] == xp.name
            ), f'xp name mismatch: {data.attrs["xp"]["name"]} != {xp.name}'

            namehash = data.attrs['calibration']['namehash']
            calib_name = f"{xp.name}_{data.attrs['calibration']['namehash']}"
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

        for calib in calibrations.values():
            for dfile in calib.data_files:
                recipe_name = dfile.attrs['sample']['recipe']
                assert recipe_name in recipe_lookup, f"Recipe {recipe_name} not found"
                dfile.recipe = recipe_lookup[recipe_name]
                assert id(dfile.recipe.data_files[-1]) == id(
                    dfile
                ), "Recipe data_files not linked correctly"

    return calibrations


total_recipes = 0
for xp in experiments.values():
    xp.recipes = xp.find_recipes(path_prefix=base_dir, recipe_subpath=RECIPE_RELATIVE_PATH)
    find_calibrated_data(xp)
    total_recipes += len(xp.recipes)

logger.info(f'found {total_recipes} recipes')

progress = tqdm(total=total_recipes)
for xp in experiments.values():
    # now we can build the networks for each recipe:
    progress.set_description(f'{xp.name}')
    for recipe in xp.recipes:
        networks = recipe.build_networks(
            lib,
            inverse='all',
            use_cache=config.paths.cache.networks,
            add_to_self=True,
        )
        progress.update(1)
progress.close()


##────────────────────────────────────────────────────────────────────────────}}}##

## {{{                   --     connect and backup db     --


def get_db_hash(db_path: str | Path) -> str:
    import sqlite3
    import xxhash

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    db_full_dump = '\n'.join(conn.iterdump())
    conn.close()
    return xxhash.xxh128(db_full_dump).hexdigest()


def backup_db_if_changed(db_path=config.db.sqlite.path, db_backup_dir=config.db.sqlite.backup.dir):
    import shutil

    db_path = Path(db_path).expanduser().resolve()
    db_backup_dir = Path(db_backup_dir).expanduser().resolve()

    db_backup_dir.mkdir(parents=True, exist_ok=True)
    existing_backups = sorted(db_backup_dir.glob(f'{db_path.stem}_*.sqlite'))
    latest_backup = None if not existing_backups else existing_backups[-1]

    new_db_backup_path = (
        db_backup_dir / f'{db_path.stem}_{time.strftime("%Y-%m-%d_%Hh%Mm%Ss")}.sqlite'
    )

    if latest_backup is not None:
        latest_backup_hash = get_db_hash(latest_backup)
        current_hash = get_db_hash(db_path)
        if latest_backup_hash == current_hash:
            logger.info('no changes in db, skipping backup')
            return

    logger.info(f'backing up db to {new_db_backup_path}')
    shutil.copy(db_path, new_db_backup_path)


# check if exists:
if not Path(config.db.sqlite.path).exists():
    logger.warn(f"db file {config.db.sqlite.path} not found. Creating new db")
    md.create_biocompdb_sqlite(config.db.sqlite.path, echo=True)

backup_db_if_changed()
engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, True)

with Session(engine) as session:
    for xp in experiments.values():
        session.add(xp)
    session.commit()


##────────────────────────────────────────────────────────────────────────────}}}
