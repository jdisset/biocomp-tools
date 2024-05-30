## {{{                          --     import     --
import hydra
from omegaconf import OmegaConf
from hydra import compose, initialize, initialize_config_dir
from hydra.core.plugins import Plugins
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.global_hydra import GlobalHydra

from pathlib import Path

##────────────────────────────────────────────────────────────────────────────}}}


class BiocompSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="biocomptools", path="pkg://biocomptools/configs")


Plugins.instance().register(BiocompSearchPathPlugin)


def reset_hydra(config_dir=None):
    GlobalHydra.instance().clear()
    if config_dir is not None:
        config_dir_path = Path(config_dir).expanduser().resolve().absolute()
        print(f'Initializing hydra with config dir {config_dir_path}')
        # make absolute:
        assert config_dir_path.exists()
        assert config_dir_path.is_dir()
        initialize_config_dir(config_dir=config_dir, version_base="1.3")
    else:
        initialize(version_base="1.3")


def load_hydra_config(cfgname: str, config_dir=None, without_parent_context=True, resolved=False):
    reset_hydra(config_dir=config_dir)
    cfg = compose(config_name=cfgname)

    if without_parent_context:
        # if we have a cfgname that's actually a path, the final config will be nested
        # in a parent context. This is not always what we want.
        ppath = cfgname.split('/')
        if len(ppath) > 1:
            for p in ppath[:-1]:
                cfg = cfg[p]
        cfg = OmegaConf.create(cfg)

    if resolved:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(cfg, dict)

    return cfg


def load_hydra_config_file(cfg_file: str | Path, config_dir=None, use_containing_dir=True):
    cfg_file_path = Path(cfg_file).expanduser().resolve().absolute()
    if not cfg_file_path.exists():
        raise ValueError(f'Config file {cfg_file_path} does not exist')
    file_dir = Path(cfg_file_path).parent.resolve().absolute().as_posix()

    cfgdir = file_dir if use_containing_dir else config_dir

    return load_hydra_config(cfgname=str(cfg_file_path.stem), config_dir=cfgdir)
