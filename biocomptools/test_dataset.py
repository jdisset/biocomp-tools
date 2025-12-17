## {{{                          --     imports     --
import os

# set BCTOOLS_DEBUG to 1 to enable debug logging
# os.environ.setdefault('BCTOOLS_DEBUG', '1')

from biocomptools.logging_config import get_logger, setup_logging
import biocomp.utils as ut
from biocomp.network import generate_network_info
from biocomptools.modelmodel import BiocompModel
from typing import TypeVar
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from pathlib import Path
from dracon.deferred import DeferredNode
from typing import Optional, Annotated
from pydantic import Field, BaseModel, ConfigDict
from biocomptools.toollib.common import config, make_context_from_types
from dracon.commandline import make_program, Arg
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from biocomptools.toollib.datasources import DataSource, DBSource
from biocomp.library import PartsLibrary

from sqlmodel import Session
from biocomp.utils import PartialFunction

from biocomp.library import load_lib

from biocomptools.toollib.networkselector import (
    build_data_manager,
    NetworkSelector,
    Regex,
    NetworkDataPair,
    NetworkSetUnion,
    NetworkSetIntersection,
    NetworkSetDifference,
    NetworkFilter,
    NetworkSet,
    UorfFilter,
)

from biocomptools.plot import DEFAULT_TYPES

import biocomptools.toollib.models as md

setup_logging(force=False)
logger = get_logger(__name__)


##────────────────────────────────────────────────────────────────────────────}}}


T = TypeVar('T')
MaybeDeferred = DeferredNode[T] | T


class DatasetTester(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataset: Annotated[NetworkSet, Arg(help='Networks in training set', short='d', is_file=True)]

    _lib: Optional[PartsLibrary] = None

    @property
    def db_session(self):
        return Session(self._engine)

    @property
    def path_prefix(self):
        return Path(config.paths.root).expanduser().resolve()

    @property
    def parts_library(self):
        assert self._lib, "Library not loaded"
        return self._lib

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self._lib = load_lib()
        self._engine = md.get_biocompdb_sqlite_engine(config.db.sqlite.path)

    def run(self):
        with self.db_session as session:
            self.dataset.run_selectors(session)

        for n in self.dataset.content:
            assert isinstance(n, NetworkDataPair)

        logger.info(f"Selectors yielded a total of {len(self.dataset)} networks")

        training_dman = build_data_manager(
            lib=self.parts_library,
            db_session=self.db_session,
            path_prefix=self.path_prefix,
            dataset=self.dataset,
        )
        print(f"Successfully loaded {len(training_dman._networks)} networks:")
        pretty_print_networks(training_dman)


def pretty_print_networks(training_dman):
    raw_X = training_dman._raw_X
    networks = training_dman._networks
    data = [
        {
            "name": network.name,
            "file": network.metadata["data_file"],
            "n_inputs": data.shape[1],
            "n_points": len(data),
            "network": network,
            'calibration_name': network.metadata.get('calibration_name', ''),
            'recipe_name': network.metadata.get('recipe_name', ''),
            'datafile': network.metadata.get('data_file', ''),
            "info": generate_network_info(network),
        }
        for data, network in zip(raw_X, networks)
    ]
    df = pd.DataFrame(data)
    console = Console()

    # Get architecture for each network
    def get_architecture_key(row):
        network_info = row['info']
        arch = network_info['architecture']
        ern_names = network_info['ern_names']
        if arch == '':
            return 'No ERN'
        else:
            return f"{arch} with {ern_names}"

    df['architecture_key'] = df.apply(get_architecture_key, axis=1)

    arch_grouped = df.groupby('architecture_key')

    for arch, arch_group in arch_grouped:
        n_nets = len(arch_group)
        arch_header = Text(f"Architecture: {arch} ({n_nets} networks)", style="#f4a261 bold")

        arch_content_parts = []

        recipe_grouped = arch_group.groupby('recipe_name')

        for recipe, group in recipe_grouped:
            if len(group) > 0:
                first_row = group.iloc[0]
                n_inputs = first_row['n_inputs']
                n_points = first_row['n_points']

            file_header = Text(f"{recipe}", style="#a8dadc bold")

            info_line = f"{n_inputs} inputs | {n_points} datapoints"

            subgroup_parts = []

            subgroup_parts.append(Align.center(file_header))
            subgroup_parts.append(Align.center(Text(info_line, style="#a8dadc")))
            # subgroup_parts.append(Text(""))  # Spacing

            network_names = Text()
            for i, (_, row) in enumerate(group.iterrows()):
                info = dict(row['info'])
                extra_info = ''
                if info.get('ern_names'):
                    extra_info += f"  -- ERNs: {', '.join(info['ern_names'])}\n"
                if info.get('cotx_str'):
                    cotx_str = info['cotx_str'].replace('\n', ', ')
                    extra_info += f"  -- {cotx_str}\n"
                if info.get('uorf_names'):
                    extra_info += f"  -- uORFs: {', '.join(info['uorf_names'])}\n"
                if info.get('markers'):
                    extra_info += f"  -- input markers: {', '.join(info['markers'])}\n"
                if info.get('dependent_outputs'):
                    extra_info += f"  -- dependent output: {', '.join(info['dependent_outputs'])}\n"

                calib_name = row['calibration_name']
                if calib_name:
                    extra_info += f"  -- calib: {calib_name}\n"

                if row['datafile']:
                    extra_info += f"  -- datafile: {row['datafile']}\n"

                # extra_info += '\n' + str(info)

                name_duplicate = []
                # find if the name exists in any other row of the df
                for _, row2 in df.iterrows():
                    if row['name'] == row2['name']:
                        name_duplicate.append(row2['name'])
                network_names.append(f"• {row['name']}", style="#2a9d8f bold")
                if len(name_duplicate) > 1:
                    network_names.append(
                        f" !! FOUND {len(name_duplicate)} INSTANCES !!", style="#e63946"
                    )

                network_names.append(f"\n{extra_info}", style="#3187A2")
                if i < len(group) - 1:
                    network_names.append("\n")

            subgroup_parts.append(network_names)

            subgroup_content = Group(*subgroup_parts)

            arch_content_parts.append(subgroup_content)
            arch_content_parts.append(Text(""))  # Add spacing between file groups

        arch_content = Group(*arch_content_parts)
        arch_panel = Panel(
            arch_content,
            title=arch_header,
            border_style="#264653",
            padding=(1, 2),
            title_align="center",
        )

        console.print(arch_panel)
        console.print()  # Add spacing between architecture panels


def main():
    cliprog = make_program(
        DatasetTester,
        name='biocomp-data-test',
        description='Test correct construction of datasets.',
    )
    testprog, _ = cliprog.parse_args(
        sys.argv[1:],
        context={
            **make_context_from_types(DEFAULT_TYPES),
            'BIOCOMP_ROOT': Path(config.paths.root).expanduser().resolve(),
        },
    )

    assert isinstance(testprog, DatasetTester)

    testprog.run()


if __name__ == '__main__':
    main()
