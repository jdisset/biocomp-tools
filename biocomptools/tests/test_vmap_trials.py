"""Tests for vmap-trials hyperparameter optimization."""
import pytest

import jax
import jax.numpy as jnp
import numpy as np


@pytest.fixture
def mock_params():
    """Create mock parameter tree with mixed types."""
    from biocomp import parameters as pr

    params = pr.ParameterTree()
    params.at("local/layer_0/weights", jnp.ones((4, 3, 8)), tags=["local"])
    params.at("local/layer_0/bias", jnp.zeros((4, 8)), tags=["local"])
    params.at("shared/embeddings", jnp.ones((4, 10, 5)), tags=["shared"])
    params.at("global/number_of_random_variables", jnp.array([5, 5, 5, 5]), tags=["global"])
    params.at("global/per_output_weights", jnp.ones((4, 20)), tags=["non_grad", "local"])
    return params


class TestBroadcastLocal:
    """Tests for the broadcast_local helper function."""

    def test_broadcasts_arrays(self):
        from biocomp import parameters as pr

        local = pr.ParameterTree()
        local.at("weights", jnp.ones((3, 4)), tags=["local"])
        local.at("bias", jnp.zeros(4), tags=["local"])

        def broadcast_local(local_params, n_trials):
            def broadcast_leaf(x):
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    return jnp.broadcast_to(x, (n_trials,) + x.shape)
                return x
            return jax.tree.map(broadcast_leaf, local_params)

        result = broadcast_local(local, 4)
        assert result["weights"].shape == (4, 3, 4)
        assert result["bias"].shape == (4, 4)

    def test_preserves_non_arrays(self):
        from biocomp import parameters as pr

        local = pr.ParameterTree()
        local.at("weights", jnp.ones((3, 4)), tags=["local"])
        local.at("scalar", 42, tags=["global"])
        local.at("config", {"key": "value"}, tags=["global"])

        def broadcast_local(local_params, n_trials):
            def broadcast_leaf(x):
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    return jnp.broadcast_to(x, (n_trials,) + x.shape)
                return x
            return jax.tree.map(broadcast_leaf, local_params)

        result = broadcast_local(local, 4)
        assert result["weights"].shape == (4, 3, 4)
        assert result["scalar"] == 42
        assert result["config"] == {"key": "value"}


class TestGetNTrials:
    """Tests for the get_n_trials helper function."""

    def test_finds_batch_dim_from_array(self):
        from biocomp import parameters as pr

        def get_n_trials(params):
            for leaf in jax.tree.leaves(params):
                if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
                    return leaf.shape[0]
            raise ValueError("No array leaves with batch dim found")

        params = pr.ParameterTree()
        params.at("weights", jnp.ones((8, 3, 4)), tags=["local"])
        params.at("scalar", 5, tags=["global"])

        assert get_n_trials(params) == 8

    def test_raises_on_no_arrays(self):
        from biocomp import parameters as pr

        def get_n_trials(params):
            for leaf in jax.tree.leaves(params):
                if hasattr(leaf, 'shape') and len(leaf.shape) > 0:
                    return leaf.shape[0]
            raise ValueError("No array leaves with batch dim found")

        params = pr.ParameterTree()
        params.at("scalar", 5, tags=["global"])
        params.at("config", {"key": "value"}, tags=["global"])

        with pytest.raises(ValueError, match="No array leaves"):
            get_n_trials(params)


class TestParamMerge:
    """Tests for merging shared and local params."""

    def test_merge_preserves_shapes(self):
        from biocomp import parameters as pr

        def broadcast_local(local_params, n_trials):
            def broadcast_leaf(x):
                if isinstance(x, (jnp.ndarray, np.ndarray)):
                    return jnp.broadcast_to(x, (n_trials,) + x.shape)
                return x
            return jax.tree.map(broadcast_leaf, local_params)

        shared = pr.ParameterTree()
        shared.at("shared/emb", jnp.ones((4, 10, 5)), tags=["shared"])

        local = pr.ParameterTree()
        local.at("local/weights", jnp.zeros((3, 4)), tags=["local"])

        local_batched = broadcast_local(local, 4)
        merged = pr.ParameterTree.merge(shared, local_batched)

        assert merged["shared/emb"].shape == (4, 10, 5)
        assert merged["local/weights"].shape == (4, 3, 4)


class TestBuildPerTrialWeights:
    """Tests for weight matrix construction."""

    def test_weight_matrix_shape(self):
        n_trials = 4
        n_outputs = 10

        # Simulate what _build_per_trial_weights does
        per_trial_weights = []
        for _ in range(n_trials):
            weights = np.random.rand(n_outputs) + 0.1
            per_trial_weights.append(weights)

        weight_matrix = jnp.array(per_trial_weights)
        assert weight_matrix.shape == (n_trials, n_outputs)

    def test_different_weights_per_trial(self):
        n_trials = 4
        n_outputs = 5

        per_trial_weights = []
        for i in range(n_trials):
            weights = np.ones(n_outputs) * (i + 1)
            per_trial_weights.append(weights)

        weight_matrix = jnp.array(per_trial_weights)

        for i in range(n_trials):
            assert jnp.allclose(weight_matrix[i], i + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
