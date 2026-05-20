# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import dracon as dr


def load_config():
    """Load the dracon config"""
    config = dr.load(
        'pkg:biocomptools:configs/default',
        enable_interpolation=True,
        # raw_dict=True,
    )
    dr.resolve_all_lazy(config)
    return config


config = load_config()
