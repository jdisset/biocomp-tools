### {{{                          --     imports     --
import sys

import openpyxl
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from biocomp import utils as ut
import json
import biocomp.datautils as du
import biocomp.plotutils as pu
import biocomp.utils as ut
import biocomp.train as train
import biocomp.compute as cmp
import biocomp.parameters as pm
import biocomp as bc
import time
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json5

# pretty print from rich
from rich import print as rprint
import argparse
import json
from pathlib import Path
from rich import print as rprint

# import common as cm
# from biocomp.datautils import DEFAULT_DATA_CONFIG

from pydantic import BaseModel

import logging
from biocomptools.toollib import common as cm
from biocomptools.toollib import plot as pl

from omegaconf import OmegaConf

##────────────────────────────────────────────────────────────────────────────}}}

config = cm.load_config()

config.db.sqlite
prog = cm.CLIProgram()
logger = logging.getLogger('build_xp_table')
logger.setLevel(logging.DEBUG)
logging.getLogger('biocomp').setLevel(logging.ERROR)

DEFAULT_CALIB_PATHS = list(config.calib.paths)
DEFAULT_CALIB_NAMES = list(config.calib.names)

DEFAULT_XP_PATH = ut.DEFAULT_XP_PATH
DEFAULT_RECIPE_PATH = ut.DEFAULT_RECIPE_PATH
DEFAULT_XP_CACHE_DIR = config.paths.cache.xp

BIOCOMP_ROOT = config.paths.root

config.paths.cache

### {{{                --     arg declaration and parsing     --

# arguments:
prog.add_argument('--calib_paths', type=str, nargs='+', default=DEFAULT_CALIB_PATHS)
prog.add_argument('--calib_names', type=str, nargs='+', default=DEFAULT_CALIB_NAMES)
prog.add_argument('--xp_path', type=str, default=DEFAULT_XP_PATH)
# --xp_path: path to the experiment files, or empty to use env default
prog.add_argument('--recipe_paths', type=str, nargs='+', default=DEFAULT_RECIPE_PATH)
prog.add_argument('--xp_cache_dir', type=str, default=DEFAULT_XP_CACHE_DIR)
prog.add_argument('--base_dir', type=str, default=BIOCOMP_ROOT)

# verbosity level
prog.add_argument('--verbose', type=int, default=0)
prog.parse_args([])
##────────────────────────────────────────────────────────────────────────────}}}
### {{{                    --     arg postprocessing     --

prog.xp_path = Path(prog.xp_path)
prog.base_dir = Path(prog.base_dir)
prog.recipe_paths = [Path(p) for p in prog.recipe_paths]
prog.lib = ut.load_lib()

assert len(prog.calib_paths) == len(prog.calib_names)

# loggers = [logging.getLogger(name) for name in sorted(logging.root.manager.loggerDict)]
logging.getLogger('jax').setLevel(logging.WARNING)
# completely silence biocomp's logger (including warning and error messages)
logging.getLogger('biocomp').setLevel(logging.CRITICAL)

# rich console
from rich.console import Console

prog.console = Console()


##────────────────────────────────────────────────────────────────────────────}}}
### {{{                  --     list all xps in experiment folder    --

xp_entries = {}
xp_objs = {}

import time

xp_folders = sorted([f for f in prog.xp_path.iterdir() if f.is_dir()])
for xp_dir in tqdm(xp_folders, desc='loading experiments'):
    warning_msg = ''
    subfolders = sorted([f for f in xp_dir.iterdir() if f.is_dir()])
    # check if there is {xp_dir.name}.xp.json5
    xp_json = xp_dir / f'{xp_dir.name}.xp.json5'
    if not xp_json.exists():
        logger.warning(f'no xp.json5 file found in {xp_dir.name}')
        continue

    base_xp_path = xp_dir.parent

    new_xp = bc.XP(
        xp_dir.name,
        base_xp_path,
        recipe_path=prog.recipe_paths,
        lib=prog.lib,
        data_path=prog.calib_paths,
        load_data=False,
        ignore_errors=True,
        show_progress=False,
    )

    recipe_loading_errors = new_xp.recipe_loading_errors
    xp_entries[new_xp.name] = {
        'name': new_xp.name,
        'transfection_date': new_xp.transfection_date,
        'path': Path(xp_dir).relative_to(prog.base_dir),
        'recipe_errors': new_xp.recipe_loading_errors,
    }
    xp_objs[new_xp.name] = new_xp


logger.info(f'found {len(xp_entries)} experiments')


##────────────────────────────────────────────────────────────────────────────}}}
### {{{            --     initial xpdf with calibration info     --


def calibration_info(xppath, calib_paths=prog.calib_paths, calib_names=prog.calib_names):
    # calib_folders = list(xppath.glob('data/calibrated_data*'))
    calib_folders = [xppath / p for p in calib_paths]
    calib_type = 'no'
    calib_plot = False
    calib_path = None
    for calib_folder, calib_name in zip(calib_folders, calib_names):
        if calib_folder.exists():
            calib_type = calib_name
            calib_path = calib_folder
            break
    # check if there is a calibration plot
    calib_diag_path = xppath / 'data' / 'unmixing_diagnostics'
    if calib_diag_path.exists():
        calib_plot = True
    return calib_type, calib_plot, calib_path


for new_xp in xp_entries.values():
    calib_type, calib_plot, _ = calibration_info(BIOCOMP_ROOT / new_xp['path'])
    new_xp['calibration_version'] = calib_type
    new_xp['has_calibration_diagnostics'] = calib_plot


##────────────────────────────────────────────────────────────────────────────}}}
### {{{  --     build networks    --
all_networks = []

total_samples = sum([len(x.samples) for x in xp_objs.values()])
logger.info(f'Building networks for {total_samples} samples')
progress = tqdm(total=total_samples, desc='Building networks')

for xpname, new_xp in list(xp_objs.items())[:]:
    is_ok = True
    progress.set_description(f'Building networks for {xpname}')
    networks, sample_names = new_xp.build_networks(
        ignore_errors=True,
        inverse='all',
        use_cache=config.paths.cache.networks,
        progress_callback=lambda _: progress.update(1),
    )
    X, Y = new_xp.get_XY(networks, sample_names, ignore_errors=True)
    if new_xp.network_building_errors:
        is_ok = False
    if new_xp.data_loading_errors:
        is_ok = False
    xp_entries[xpname]['network_building_errors'] = new_xp.network_building_errors
    xp_entries[xpname]['data_loading_errors'] = new_xp.data_loading_errors
    assert len(networks) == len(X) == len(Y)
    for i, net_entry in enumerate(networks):
        if net_entry:
            sname = sample_names[i]
            data_file = new_xp.get_sample_data_file(sname, ignore_errors=True)
            # subtract prog.xp_path from data_file to get the relative path:
            if data_file is not None:
                data_file = Path(data_file).relative_to(prog.base_dir)

            recipe_file = net_entry.metadata['recipe_file']
            if recipe_file is not None:
                recipe_file = Path(recipe_file).relative_to(prog.base_dir)

            net_entry = {
                'xp': xpname,
                'network': net_entry,
                'sample_name': sname,
                'recipe_name': net_entry.metadata['recipe_name'],
                'recipe_file': recipe_file,
                'data_file': data_file,
                # 'network_info': net_entry.metadata.get('network_info', None),
                'network_info': bc.network.generate_network_info(net_entry),
            }
            all_networks.append(net_entry)

    if is_ok:
        logger.info(f'checking data for {xpname}')
        for x, y, net_entry in zip(X, Y, networks):
            if x is None or y is None or x.size == 0 or y.size == 0:
                is_ok = False
                xp_entries[xpname][
                    'data_loadng_errors'
                ] += f'empty data for network {net_entry.name}\n\n'


##────────────────────────────────────────────────────────────────────────────}}}##

## {{{                   --     connect and backup db     --

from copy import deepcopy
import sqlite3
import xxhash
import biocomptools.toollib.models as md
from typing import TypeVar, List
import shutil
import difflib
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.inspection import inspect
from sqlmodel import tuple_
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Type

M = TypeVar('M', bound=BaseModel)
SQL_M = TypeVar('SQL_M', bound=SQLModel)


def get_db_hash(db_path: str | Path) -> str:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    db_full_dump = '\n'.join(conn.iterdump())
    conn.close()
    return xxhash.xxh128(db_full_dump).hexdigest()


def backup_db_if_changed(db_path=config.db.sqlite.path, db_backup_dir=config.db.sqlite.backup.dir):
    db_path = Path(db_path)
    db_backup_dir = Path(db_backup_dir)

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
            logger.info(f'no changes in db, skipping backup')
            return

    logger.info(f'backing up db to {new_db_backup_path}')
    shutil.copy(db_path, new_db_backup_path)


def diff_strings(a: str, b: str) -> str:
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    ADD_COLOR = '\x1b[38;5;16;48;5;78m'
    DEL_COLOR = '\x1b[38;5;16;48;5;210m'
    END_ADD = '\x1b[0m'
    END_DEL = '\x1b[0m'
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(a[a0:a1])
        elif opcode == 'insert':
            output.append(f'{ADD_COLOR}{b[b0:b1]}{END_ADD}')
        elif opcode == 'delete':
            output.append(f'{DEL_COLOR}{a[a0:a1]}{END_DEL}')
        elif opcode == 'replace':
            output.append(f'{ADD_COLOR}{b[b0:b1]}{END_ADD}')
            output.append(f'{DEL_COLOR}{a[a0:a1]}{END_DEL}')
    return ''.join(output)


def merge_model(mA: M, mB: M, priorities_A=[], except_if_null=True):
    """Merge B into A, keeping the values from A if they are in priorities_A,
    and from B otherwise. If except_if_null is True, then values from B that are None are not copied
    """
    ma_dump = mA.model_dump()
    for k, v in mB.model_dump().items():
        if k in priorities_A:
            continue
        if except_if_null and v is None:
            continue
        ma_dump[k] = v
    mA.model_validate(ma_dump)


def print_model_diff(mA: M, mB: M):
    """Print the differences between two models"""
    ma_dump = mA.model_dump()
    mb_dump = mB.model_dump()
    for k in ma_dump.keys():
        if k not in mb_dump:
            print(f'Model B does not have key {k}')
        elif ma_dump[k] != mb_dump[k]:
            print(f'{k}: {diff_strings(str(ma_dump[k]), str(mb_dump[k]))}')
    for k in mb_dump.keys():
        if k not in ma_dump:
            print(f'Model A does not have key {k}')


def update_records(new_records: List[SQL_M], merge_keep: list[str] = []):
    model_type = type(new_records[0])
    engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, False)
    backup_db_if_changed()

    keys = inspect(model_type).primary_key

    def get_key(record):
        return tuple(getattr(record, k.name) for k in keys)

    with Session(engine) as session:
        for new_record in new_records:
            try:
                cur_record = session.exec(
                    select(model_type).where(tuple_(*get_key(model_type)) == get_key(new_record))
                ).one()
            except NoResultFound:
                rprint(f'[bold green]adding new record {new_record}[/bold green]')
                session.add(new_record)
                session.commit()
            else:
                old_record = deepcopy(cur_record)
                merge_model(cur_record, new_record, priorities_A=merge_keep)
                if cur_record == old_record:
                    rprint(f'[bold blue]{get_key(new_record)}[/bold blue] has no changes')
                else:
                    rprint(f'[bold yellow]updating record {new_record}[/bold yellow]')
                    print_model_diff(old_record, cur_record)
                    session.add(cur_record)
                    session.commit()
                    session.refresh(cur_record)


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     create and update xpdf     --

xpdf = pd.DataFrame(xp_entries).T

# replace all the *_errors types to string

error_cols = sorted([col for col in xpdf.columns if '_errors' in col])
for col in error_cols:
    xpdf[col] = xpdf[col].astype(str)
    xpdf[col] = xpdf[col].apply(lambda x: x.replace('nan', ''))

all_xps = [md.Experiment(**row.to_dict()) for i, row in xpdf.iterrows()]

# now we can update all the xps
backup_db_if_changed()
update_records(all_xps, merge_keep=['comments'])


##────────────────────────────────────────────────────────────────────────────}}}

### {{{                  --     create and update netdf     --

netdf = pd.DataFrame(all_networks)

all_networks = [
    md.Network.model_validate({'name': '', **row.to_dict()}) for i, row in netdf.iterrows()
]
all_names = set()
for n in all_networks:
    if n.network_info is None:
        print(f'network {n.name} has no network_info')
        print(n)
    n.name = n.generate_unique_name()
    if n.name in all_names:
        raise ValueError(f'duplicate name {n.name}')
    all_names.add(n.name)
    if type(n.recipe_file) is not str:
        print(f'network {n.name} has recipe path of type {type(n.recipe_file)}')

##
backup_db_if_changed()

# clear the entire table
from sqlalchemy import delete
engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path, False)
with Session(engine) as session:
    session.exec(delete(md.Network))
    session.commit()


update_records(all_networks, merge_keep=['data_quality', 'comments'])


##────────────────────────────────────────────────────────────────────────────}}}
