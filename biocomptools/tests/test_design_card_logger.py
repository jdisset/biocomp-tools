# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
import json
import pickle
from dataclasses import dataclass

import jax
import numpy as np

from biocomptools.logger_history import LoggerContext
from biocomptools.toollib.loggers.designcardlogger import DesignCardLogger, SingleNetworkReplayState


class FakeNetwork:
    def __init__(self, name: str):
        self.name = name

    def model_copy(self, deep: bool = False):
        return FakeNetwork(self.name)


class FakeDManager:
    def __init__(self, networks):
        self.networks = networks
        self.targets = []
        self.enable_tu_masking = False
        self._stack = None

    def model_copy(self, update=None):
        update = update or {}
        return FakeDManager(update.get("networks", list(self.networks)))

    def build_stack(self, model):
        assert self._stack is not None
        return self._stack


class FakeDConfig:
    def __init__(self):
        self.hard_pruning_enabled = True
        self.hard_pruning_interval = 10
        self.n_replicates = 1
        self.hard_pruning_ratio_threshold = 0.01
        self.hard_pruning_preserve_minimum_tus = 1
        self.hard_pruning_prune_margin = 0.1
        self.auto_lock_topology_tus = False
        self.enable_tu_masking = False
        self.hard_pruning_top_percent = None
        self.hard_pruning_min_networks = None
        self.use_latent_ratios = False
        self.latent_dim = 8
        self.latent_hidden_dim = 16
        self.seed_key = jax.random.PRNGKey(0)

    def model_copy(self, update=None):
        clone = FakeDConfig()
        clone.__dict__.update(self.__dict__)
        if update:
            clone.__dict__.update(update)
        return clone


class FakeDB:
    def __init__(self, dmanager, dconfig, model, blobs):
        self._artifacts = {
            "dmanager": dmanager,
            "dconfig": dconfig,
            "model": model,
        }
        self._blobs = blobs

    def load_artifact(self, name):
        return self._artifacts.get(name)

    def get_step_range(self):
        return (1, 20)

    def load_blob(self, step, key):
        return self._blobs.get((step, key))


class FakeStack:
    def __init__(self, label, committed_names):
        self.label = label
        self._committed_names = committed_names
        self.committed_params = []

    def commit(self, params):
        self.committed_params.append(params)
        return [FakeNetwork(name) for name in self._committed_names]


class ShapeCheckingStack(FakeStack):
    def commit(self, params):
        self.committed_params.append(params)
        assert np.asarray(params["token"]).shape == (3, 2)
        return [FakeNetwork(name) for name in self._committed_names]


class PruningAwareStack(FakeStack):
    def commit(self, params):
        self.committed_params.append(params)
        assert params["pruned"] is True
        return [FakeNetwork(name) for name in self._committed_names]


@dataclass
class FakeSnapshot:
    loss: np.ndarray


def _write_metadata(run_dir, *, n_replicates=2):
    meta = {
        "design_info": {
            "network_names": ["netA", "netB"],
            "n_replicates": n_replicates,
        }
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta))


def test_track_recipe_hash_resolves_best_design(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "replay_cards"
    out_dir.mkdir(parents=True)
    _write_metadata(run_dir)

    best_designs = {
        "target0": {
            "recipe_hash": "winner-hash",
            "network_name": "netA_rep1",
            "network_id": 2,
            "replicate": 1,
            "target_id": 0,
            "target": None,
        }
    }
    with open(run_dir / "best_designs.pickle", "wb") as f:
        pickle.dump(best_designs, f)

    logger = DesignCardLogger(output_dir=str(out_dir), track_recipe_hash="winner-hash")
    logger.initialize(None)

    assert logger._original_flat_index == 2
    assert logger._resolved_name == "netA_rep1"
    assert logger._resolved_replicate == 1


def test_hard_prune_replay_commits_against_segment_stack(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "replay_cards"
    out_dir.mkdir(parents=True)
    _write_metadata(run_dir, n_replicates=1)

    initial_dmanager = FakeDManager([FakeNetwork("netA"), FakeNetwork("netB")])
    initial_stack = FakeStack("initial", ["commit-initial-A", "commit-initial-B"])
    post_dmanager = FakeDManager([FakeNetwork("netB")])
    post_stack = FakeStack("post", ["commit-post-B"])
    dconfig = FakeDConfig()
    model = object()

    early_params = {"token": np.array([[5]])}
    boundary_params = {"token": np.array([[10]])}
    post_params = {"token": np.array([[11]])}
    db = FakeDB(
        initial_dmanager,
        dconfig,
        model,
        {
            (5, "latest_params"): early_params,
            (10, "latest_params"): boundary_params,
            (11, "latest_params"): post_params,
        },
    )

    def fake_build_stack_from_dconf(dmanager, dconf, model, lock_ratios=False):
        assert dmanager is initial_dmanager
        return initial_stack

    def fake_identify_tus_to_prune(*args, **kwargs):
        return {0: set(), 1: {"tu1"}}

    def fake_evaluate_segment_snapshot(*args, **kwargs):
        return FakeSnapshot(loss=np.array([0.6, 0.1]))

    def fake_hard_prune_and_rebuild(*args, **kwargs):
        return post_dmanager, post_stack, {"token": np.array(10)}

    monkeypatch.setattr(
        "biocomp.design_prune_controller.build_stack_from_dconf",
        fake_build_stack_from_dconf,
    )
    monkeypatch.setattr(
        "biocomp.design_prune_controller.evaluate_segment_snapshot",
        fake_evaluate_segment_snapshot,
    )
    monkeypatch.setattr(
        "biocomp.design_pruning.identify_tus_to_prune",
        fake_identify_tus_to_prune,
    )
    monkeypatch.setattr(
        "biocomp.design_pruning.hard_prune_and_rebuild",
        fake_hard_prune_and_rebuild,
    )

    logger = DesignCardLogger(output_dir=str(out_dir), network_id=1)
    logger.initialize(None)
    context = LoggerContext.build(step=11, output_dir=out_dir, db=db)
    logger._ensure_context_init(context)

    before_net, before_idx = logger._get_committed_network_at_step(5)
    after_net, after_idx = logger._get_committed_network_at_step(11)

    assert before_idx == 1
    assert before_net.name == "commit-initial-B"
    assert after_idx == 0
    assert after_net.name == "commit-post-B"
    assert np.asarray(initial_stack.committed_params[0]["token"]).item() == 5
    assert np.asarray(post_stack.committed_params[0]["token"]).item() == 11


def test_replay_commit_strips_extra_leading_singleton_axis(tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "replay_cards"
    out_dir.mkdir(parents=True)
    _write_metadata(run_dir, n_replicates=1)

    dmanager = FakeDManager([FakeNetwork("netA"), FakeNetwork("netB")])
    stack = ShapeCheckingStack("initial", ["commit-A", "commit-B"])
    dmanager._stack = stack
    dconfig = FakeDConfig()
    dconfig.hard_pruning_enabled = False
    model = object()

    params = {
        "token": np.arange(6).reshape(1, 1, 3, 2),
    }
    db = FakeDB(
        dmanager,
        dconfig,
        model,
        {
            (5, "latest_params"): params,
        },
    )

    logger = DesignCardLogger(output_dir=str(out_dir), network_id=1)
    logger.initialize(None)
    context = LoggerContext.build(step=5, output_dir=out_dir, db=db)
    context.dmanager = dmanager
    context.model = model
    logger._ensure_context_init(context)
    logger._replay_states[0].stack = stack

    net, idx = logger._get_committed_network_at_step(5)

    assert idx == 1
    assert net.name == "commit-B"


def test_apply_pruning_rules_on_commit_uses_single_network_pruning(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    out_dir = run_dir / "replay_cards"
    out_dir.mkdir(parents=True)
    _write_metadata(run_dir, n_replicates=1)

    dmanager = FakeDManager([FakeNetwork("netA"), FakeNetwork("netB")])
    dconfig = FakeDConfig()
    dconfig.hard_pruning_enabled = False
    model = object()
    dmanager._stack = FakeStack("initial", ["commit-A", "commit-B"])
    db = FakeDB(
        dmanager,
        dconfig,
        model,
        {
            (5, "latest_params"): {"token": np.array([[5]])},
        },
    )

    logger = DesignCardLogger(
        output_dir=str(out_dir),
        network_id=1,
        apply_pruning_rules_on_commit=True,
    )
    logger.initialize(None)
    context = LoggerContext.build(step=5, output_dir=out_dir, db=db)
    context.dmanager = dmanager
    context.model = model
    logger._ensure_context_init(context)

    prune_stack = PruningAwareStack("single", ["commit-pruned-B"])
    single_state = SingleNetworkReplayState(
        stack=prune_stack,
        dmanager=FakeDManager([FakeNetwork("netB")]),
        current_idx=1,
        params_template={"token": np.array([[5]]), "pruned": False},
        row_indices_by_namespace={},
    )

    monkeypatch.setattr(logger, "_get_single_network_state", lambda state, current_idx: single_state)
    monkeypatch.setattr(
        logger,
        "_project_params_to_single_network",
        lambda commit_params, _single_state: {"token": np.array([[5]]), "pruned": False},
    )
    monkeypatch.setattr(
        logger,
        "_apply_pruning_rules_to_single_network",
        lambda _single_state, params: {**params, "pruned": True},
    )

    net, idx = logger._get_committed_network_at_step(5)

    assert idx == 1
    assert net.name == "commit-pruned-B"
    assert prune_stack.committed_params[0]["pruned"] is True
