# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Tests for biocomp-tuner module."""

import numpy as np

from biocomptools.tuner import TunerConfig, ParamDescriptor
from biocomp.designloss import GridLossWeights
from biocomptools.tuner.param_schema import (
    is_ratio_param,
    is_embedding_param,
    get_mask_options_count,
    make_display_name,
)


class TestParamSchema:
    """Tests for parameter schema and filtering."""

    def test_is_ratio_param(self):
        assert is_ratio_param("local/layer_2/ratios")
        assert is_ratio_param("local/layer_5/ratio")
        assert not is_ratio_param("local/layer_2/tl_rate")
        assert not is_ratio_param("local/layer_2/quantization_mask/ratios")

    def test_is_embedding_param(self):
        assert is_embedding_param("local/layer_2/tl_rate")
        assert is_embedding_param("local/layer_2/embedding")
        assert is_embedding_param("local/layer_2/some_embedding")
        assert not is_embedding_param("local/layer_2/ratios")
        assert not is_embedding_param("local/layer_2/bias")

    def test_get_mask_options_count_1d(self):
        mask = np.array([True, False, True, True])
        assert get_mask_options_count(mask) == 3

    def test_get_mask_options_count_2d(self):
        mask = np.array([
            [True, False, True],
            [True, True, True],
            [False, True, False],
        ])
        assert get_mask_options_count(mask) == 3

    def test_get_mask_options_count_single_option(self):
        mask = np.array([False, True, False])
        assert get_mask_options_count(mask) == 1

    def test_make_display_name(self):
        assert make_display_name("local/layer_2/ratios") == "Layer 2 › Ratios"
        assert "Layer 5" in make_display_name("local/layer_5/tl_rate")


class TestParamDescriptor:
    """Tests for ParamDescriptor dataclass."""

    def test_to_dict(self):
        desc = ParamDescriptor(
            path="local/layer_2/ratios",
            display_name="Layer 2 Ratios",
            shape=(4, 8),
            category="ratios",
            current_value=[[0.1] * 8] * 4,
        )
        d = desc.to_dict()
        assert d["path"] == "local/layer_2/ratios"
        assert d["category"] == "ratios"
        assert d["shape"] == [4, 8]


class TestTunerConfig:
    """Tests for TunerConfig."""

    def test_default_config(self):
        config = TunerConfig()
        assert config.grid_resolution == (32, 32)
        assert config.weights.w_sinkhorn == 1.0
        assert config.weights.w_lncc == 0.5
        assert config.weights.w_mse == 1.0
        assert config.weights.w_simse == 1.0

    def test_custom_config(self):
        config = TunerConfig(
            grid_resolution=(64, 64),
            weights=GridLossWeights(
                w_sinkhorn=0.5,
                w_lncc=0.3,
                w_mse=0.2,
            ),
        )
        assert config.grid_resolution == (64, 64)
        assert config.weights.w_sinkhorn == 0.5
