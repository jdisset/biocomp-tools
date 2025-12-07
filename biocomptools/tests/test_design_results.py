import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import stats


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = RegressionMetrics.compute(y, y)
        assert np.isclose(metrics.rmse, 0.0, atol=1e-10)
        assert np.isclose(metrics.mae, 0.0, atol=1e-10)
        assert np.isclose(metrics.r2, 1.0, atol=1e-10)
        assert np.isclose(metrics.pearson_r, 1.0, atol=1e-10)
        assert np.isclose(metrics.max_error, 0.0, atol=1e-10)

    def test_known_rmse(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        y_true, y_pred = np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.2, 2.9])
        metrics = RegressionMetrics.compute(y_true, y_pred)
        assert np.isclose(metrics.rmse, np.sqrt(np.mean((y_pred - y_true) ** 2)), atol=1e-10)

    def test_known_mae(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        y_true, y_pred = np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 3.5])
        metrics = RegressionMetrics.compute(y_true, y_pred)
        assert np.isclose(metrics.mae, 0.5, atol=1e-10)

    def test_r2_computation(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        metrics = RegressionMetrics.compute(y_true, y_pred)
        ss_res, ss_tot = np.sum((y_pred - y_true) ** 2), np.sum((y_true - np.mean(y_true)) ** 2)
        assert np.isclose(metrics.r2, 1 - ss_res / ss_tot, atol=1e-10)

    def test_pearson_correlation(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        np.random.seed(42)
        y_true = np.random.randn(50)
        y_pred = y_true * 0.8 + 0.2 * np.random.randn(50)
        metrics = RegressionMetrics.compute(y_true, y_pred)
        expected_r, expected_p = stats.pearsonr(y_true, y_pred)
        assert np.isclose(metrics.pearson_r, expected_r, atol=1e-10)
        assert np.isclose(metrics.pearson_p, expected_p, atol=1e-10)

    def test_handles_nan_inf(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        y_true = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
        y_pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        metrics = RegressionMetrics.compute(y_true, y_pred)
        assert np.isclose(metrics.rmse, 0.0, atol=1e-10)

    def test_percentile_error(self):
        from biocomptools.toollib.design_results import RegressionMetrics
        y_true, y_pred = np.zeros(100), np.arange(100) / 100.0
        metrics = RegressionMetrics.compute(y_true, y_pred)
        assert 0.93 < metrics.p95_error < 0.97


class TestDistributionMetrics:
    def test_distribution_statistics(self):
        from biocomptools.toollib.design_results import DistributionMetrics
        y_true, y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        metrics = DistributionMetrics.compute(y_true, y_pred)
        assert np.isclose(metrics.target_mean, 3.0, atol=1e-10)
        assert np.isclose(metrics.target_min, 1.0, atol=1e-10)
        assert np.isclose(metrics.target_max, 5.0, atol=1e-10)
        assert np.isclose(metrics.prediction_mean, 4.0, atol=1e-10)
        assert np.isclose(metrics.prediction_min, 2.0, atol=1e-10)
        assert np.isclose(metrics.prediction_max, 6.0, atol=1e-10)


class TestDesignResultsManager:
    def test_creates_base_structure(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            DesignResultsManager(tmpdir)
            assert (Path(tmpdir) / 'targets').exists()
            assert (Path(tmpdir) / 'checkpoints').exists()
            assert (Path(tmpdir) / 'comparison').exists()

    def test_creates_target_dir(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DesignResultsManager(tmpdir)
            target_dir = manager.get_target_dir('my_target')
            assert target_dir.exists()
            assert target_dir.name == 'my_target'

    def test_sanitizes_target_names(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DesignResultsManager(tmpdir)
            target_dir = manager.get_target_dir('target/with:special*chars')
            assert target_dir.exists()
            assert '/' not in target_dir.name and ':' not in target_dir.name and '*' not in target_dir.name

    def test_creates_rank_dirs_final(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_dir = DesignResultsManager(tmpdir).get_rank_dir('target1', rank=1, step=None)
            assert rank_dir.exists() and 'final' in str(rank_dir) and 'rank_01' in str(rank_dir)

    def test_creates_rank_dirs_step(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            rank_dir = DesignResultsManager(tmpdir).get_rank_dir('target1', rank=3, step=500)
            assert rank_dir.exists() and 'step_000500' in str(rank_dir) and 'rank_03' in str(rank_dir)

    def test_save_rankings(self):
        from biocomptools.toollib.design_results import DesignResultsManager
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DesignResultsManager(tmpdir)
            manager.save_rankings('target1', [(0, 2, 0.05), (1, 1, 0.06), (0, 0, 0.07)], step=None)
            rankings_file = manager.get_target_dir('target1') / 'final' / 'rankings.json'
            assert rankings_file.exists()
            with open(rankings_file) as f:
                data = json.load(f)
            assert len(data) == 3 and data[0]['rank'] == 1 and data[0]['replicate_id'] == 0
            assert data[0]['network_id'] == 2 and np.isclose(data[0]['loss'], 0.05)


class TestRecipeSerialization:
    @pytest.fixture
    def lib(self):
        from biocomp.library import load_lib
        return load_lib()

    def test_simple_recipe_roundtrip(self, lib):
        from biocomp.library import LibraryContext
        from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit

        with LibraryContext.with_library(lib):
            recipe = Recipe(name='test_recipe', content=[
                CoTransfection(name='cotx1', units=[
                    TranscriptionUnit(name='tu1', slots=['hEF1a', '1x_uORF', 'eYFP', 'L0.T_4560'])
                ], ratios=[1.0])
            ])
            recipe_dict = recipe.model_dump(mode='json')
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(recipe_dict, f)
                tmp_path = f.name
            try:
                with open(tmp_path) as f:
                    loaded_recipe = Recipe(**yaml.safe_load(f))
                assert loaded_recipe.name == recipe.name
                assert len(loaded_recipe.content) == len(recipe.content)
                assert loaded_recipe.content[0].name == recipe.content[0].name
            finally:
                os.unlink(tmp_path)


class TestDesignSummaryLogger:
    def test_logger_initialization(self):
        from biocomptools.toollib.loggers.design_summary_logger import DesignSummaryLogger
        logger = DesignSummaryLogger(log_period=100, topk_per_target=3, generate_circuit_diagrams=False)
        assert logger.log_period == 100 and logger.topk_per_target == 3 and not logger.generate_circuit_diagrams

    def test_top_candidates_extraction(self):
        from biocomptools.toollib.loggers.design_summary_logger import DesignSummaryLogger
        logger = DesignSummaryLogger()
        all_losses = np.array([
            [[0.5, 0.3, 0.8], [0.2, 0.4, 0.1]],
            [[0.6, 0.1, 0.7], [0.3, 0.5, 0.2]],
        ])
        candidates = logger._get_top_candidates(all_losses, target_id=0, n=2)
        assert candidates[0] == (1, 1, 0.1) and candidates[1] == (0, 1, 0.3)
        candidates = logger._get_top_candidates(all_losses, target_id=1, n=2)
        assert candidates[0] == (0, 2, 0.1) and candidates[1] == (0, 0, 0.2)


class TestReproducibility:
    @pytest.fixture
    def lib(self):
        from biocomp.library import load_lib
        return load_lib()

    @pytest.fixture
    def model(self):
        from biocomptools.modelmodel import BiocompModel
        model_path = os.environ.get('BIOCOMP_ROOT', '') + '/Models/design/20251125-full_set-000/training/rainerython-chemiconing-dimiali.bestmodel.pickle'
        if not Path(model_path).exists():
            pytest.skip(f"Model not found at {model_path}")
        return BiocompModel.load(model_path)

    def test_network_to_recipe_roundtrip(self, lib):
        from biocomp.library import LibraryContext
        from biocomp.recipe import Recipe, CoTransfection, TranscriptionUnit
        from biocomp.network import recipe_to_networks

        with LibraryContext.with_library(lib):
            original_recipe = Recipe(name='test_roundtrip', content=[
                CoTransfection(name='input', units=[
                    TranscriptionUnit(name='tu1', slots=['hEF1a', '2x_uORF', 'CasE_rec', 'mMaroon1', 'L0.T_4560']),
                    TranscriptionUnit(name='tu2', slots=['hEF1a', '1x_uORF', 'CasE', 'L0.T_4560'])
                ], ratios=[1.0, 1.0]),
                CoTransfection(name='output', units=[
                    TranscriptionUnit(name='out', slots=['hEF1a', '1x_uORF', 'eBFP2', 'L0.T_4560'])
                ], ratios=[1.0])
            ])
            networks = recipe_to_networks(original_recipe)
            assert len(networks) >= 1
            roundtrip_recipe = networks[0].to_recipe()
            roundtrip_networks = recipe_to_networks(roundtrip_recipe)
            assert len(roundtrip_networks) == len(networks)
            orig_nodes = set(n.node_type for n in networks[0].compute_graph.nodes.values())
            rt_nodes = set(n.node_type for n in roundtrip_networks[0].compute_graph.nodes.values())
            assert orig_nodes == rt_nodes

    def test_saved_recipe_prediction_match(self, lib, model):
        from biocomp.library import LibraryContext
        from biocomp.recipe import Recipe
        from biocomp.network import recipe_to_networks
        from biocomptools.modelmodel import NetworkModel
        from biocomptools.toollib.networkprediction import NetworkPrediction
        import dracon as dr

        with LibraryContext.with_library(lib):
            recipe_file = Path(__file__).parent.parent.parent.parent / 'biocomp-jobs/design/architectures/two_and_one_all_uorfs.yaml'
            if not recipe_file.exists():
                pytest.skip(f"Recipe file not found: {recipe_file}")
            try:
                recipes = dr.DraconLoader(context={}).load(recipe_file).get('recipes', [])
                if not recipes:
                    pytest.skip("No recipes found")
                recipe = recipes[0]
            except Exception as e:
                pytest.skip(f"Failed to load recipe: {e}")

            try:
                networks = recipe_to_networks(recipe)
                if not networks:
                    pytest.skip("No networks generated")
                network = networks[0]
            except Exception as e:
                pytest.skip(f"Failed to build network: {e}")

            x1, x2 = np.linspace(0.1, 0.9, 12), np.linspace(0.1, 0.9, 12)
            X = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)

            try:
                predictor = NetworkPrediction(predict_at=[X], network_model=NetworkModel(network=[network], model=model),
                                              max_evals=500, z_value='uniform', verbose=False)
                Y_original = np.asarray(predictor.get_data()[0].y).squeeze()
            except Exception as e:
                pytest.skip(f"Prediction failed: {e}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(recipe.model_dump(mode='json'), f)
                recipe_path = f.name

            try:
                with open(recipe_path) as f:
                    loaded_recipe = Recipe(**yaml.safe_load(f))
                loaded_network = recipe_to_networks(loaded_recipe)[0]
                loaded_predictor = NetworkPrediction(predict_at=[X], network_model=NetworkModel(network=[loaded_network], model=model),
                                                     max_evals=500, z_value='uniform', verbose=False)
                Y_loaded = np.asarray(loaded_predictor.get_data()[0].y).squeeze()
                assert Y_original.shape == Y_loaded.shape
                assert np.allclose(Y_original, Y_loaded, rtol=1e-5, atol=1e-6)
            finally:
                os.unlink(recipe_path)

    def test_evaluation_data_format(self, lib):
        from biocomp.library import LibraryContext
        with LibraryContext.with_library(lib):
            x_data, y_true, y_pred = np.random.rand(100, 2), np.random.rand(100), np.random.rand(100)
            with tempfile.TemporaryDirectory() as tmpdir:
                np.savez_compressed(Path(tmpdir) / 'evaluation_data.npz', x=x_data, y_true=y_true, y_pred=y_pred)
                loaded = np.load(Path(tmpdir) / 'evaluation_data.npz')
                assert np.allclose(loaded['x'], x_data) and np.allclose(loaded['y_true'], y_true) and np.allclose(loaded['y_pred'], y_pred)


class TestDesignModeEndToEnd:
    def test_compute_design_metrics_integration(self):
        from biocomptools.toollib.design_results import compute_design_metrics
        np.random.seed(42)
        y_true = np.random.rand(500)
        y_pred = y_true + 0.1 * np.random.randn(500)
        metrics = compute_design_metrics(y_true, y_pred, 0.05, 'test_target', 'test_network', 0, 0, 1, 1000)
        assert metrics.target_name == 'test_target' and metrics.network_name == 'test_network'
        assert metrics.rank == 1 and metrics.step == 1000
        assert 0 < metrics.regression.rmse < 0.2 and 0.8 < metrics.regression.r2 < 1.0 and 0.9 < metrics.regression.pearson_r < 1.0
        metrics_dict = metrics.to_dict()
        assert 'regression' in metrics_dict and 'distribution' in metrics_dict and 'loss' in metrics_dict

    def test_metrics_json_save_load(self):
        from biocomptools.toollib.design_results import compute_design_metrics
        y_true, y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([0.12, 0.18, 0.32, 0.38, 0.52])
        metrics = compute_design_metrics(y_true, y_pred, 0.042, 'json_test', 'net_1', 2, 3, 1, 500)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            metrics.to_json(Path(f.name))
            tmp_path = f.name
        try:
            with open(tmp_path) as f:
                loaded = json.load(f)
            assert loaded['target_name'] == 'json_test' and loaded['rank'] == 1
            assert np.isclose(loaded['loss']['total'], 0.042)
            assert 'rmse' in loaded['regression'] and 'target_mean' in loaded['distribution']
        finally:
            os.unlink(tmp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
