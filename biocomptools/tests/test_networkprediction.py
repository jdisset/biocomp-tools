"""Simplified tests for NetworkPrediction core functionality."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from biocomptools.toollib.networkprediction import (
    NetworkStats,
    make_hypercube,
    to_array_list,
    ensure_list,
    log_shapes,
    _calculate_single_network_stats,
)


class TestNetworkStats:
    """Test the NetworkStats utility class."""
    
    def test_calculate_base_stats(self):
        """Test basic statistics calculation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        stats = NetworkStats.calculate_base_stats(data)
        
        assert stats['latent_mean'] == pytest.approx(3.5)
        assert stats['latent_std'] == pytest.approx(1.707, rel=1e-2)
        assert stats['latent_min'] == 1.0
        assert stats['latent_max'] == 6.0
    
    def test_calculate_comparison_stats(self):
        """Test comparison statistics calculation."""
        pred = np.array([[1.0, 2.0], [3.0, 4.0]])
        gt = np.array([[1.1, 2.1], [2.9, 3.9]])
        
        stats = NetworkStats.calculate_comparison_stats(pred, gt)
        
        expected_mse = np.mean((pred - gt) ** 2)
        assert stats['mse'] == pytest.approx(expected_mse)
        assert stats['rmse'] == pytest.approx(np.sqrt(expected_mse))


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_make_hypercube(self):
        """Test hypercube generation."""
        cube = make_hypercube(ndim=2, res=3, xmin=0, xmax=1)
        
        assert cube.shape == (9, 2)  # 3^2 points in 2D
        assert cube.min() == 0.0
        assert cube.max() == 1.0
    
    def test_make_hypercube_single_dim(self):
        """Test 1D hypercube."""
        cube = make_hypercube(ndim=1, res=5)
        assert cube.shape == (5, 1)
    
    @pytest.mark.parametrize("ndim,res", [(0, 10), (2, 0), (-1, 5)])
    def test_make_hypercube_invalid_params(self, ndim, res):
        """Test invalid parameters raise assertions."""
        with pytest.raises(AssertionError):
            make_hypercube(ndim=ndim, res=res)
    
    def test_to_array_list_single_array(self):
        """Test converting single array to list."""
        arr = np.array([[1, 2], [3, 4]])
        result = to_array_list(arr)
        
        assert len(result) == 1
        assert np.array_equal(result[0], arr.astype(np.float32))
    
    def test_to_array_list_already_list(self):
        """Test passing through array list."""
        arr1 = np.array([[1, 2]])
        arr2 = np.array([[3, 4]])
        input_list = [arr1, arr2]
        
        result = to_array_list(input_list)
        
        assert len(result) == 2
        assert np.array_equal(result[0], arr1.astype(np.float32))
        assert np.array_equal(result[1], arr2.astype(np.float32))
    
    def test_to_array_list_with_none(self):
        """Test handling None values."""
        input_list = [np.array([[1, 2]]), None]
        result = to_array_list(input_list, allow_none=True)
        
        assert len(result) == 2
        assert result[1] is None
    
    def test_to_array_list_none_input(self):
        """Test None input with allow_none."""
        result = to_array_list(None, allow_none=True)
        assert result is None
    
    def test_ensure_list_int(self):
        """Test converting int to list."""
        assert ensure_list(5) == [5]
    
    def test_ensure_list_already_list(self):
        """Test list passthrough."""
        assert ensure_list([1, 2, 3]) == [1, 2, 3]


class TestSingleNetworkStats:
    """Test _calculate_single_network_stats function."""
    
    @pytest.fixture
    def mock_rescaler(self):
        """Create mock rescaler."""
        rescaler = Mock()
        rescaler.fwd = Mock(side_effect=lambda x: x)  # Identity function
        return rescaler
    
    def test_calculate_stats_no_ground_truth(self, mock_rescaler):
        """Test statistics calculation without ground truth."""
        yhat = np.random.rand(100, 3).astype(np.float32)
        x = np.random.rand(100, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=None,
            x=x,
            dependent_output_pos=[0, 1],  # First two columns
            nb_points_in_eval=100,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_net'},
            enable_gridstats=False
        )
        
        assert stats['network_name'] == 'test_net'
        assert stats['samples'] == 100
        assert stats['mse'] is None
        assert stats['rmse'] is None
        assert 'latent_mean' in stats
        assert 'latent_std' in stats
    
    def test_calculate_stats_with_ground_truth(self, mock_rescaler):
        """Test statistics calculation with ground truth."""
        yhat = np.random.rand(100, 3).astype(np.float32)
        gt = np.random.rand(100, 2).astype(np.float32)  # Only dependent outputs
        x = np.random.rand(100, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=gt,
            x=x,
            dependent_output_pos=[0, 1],  # First two columns
            nb_points_in_eval=100,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_net'},
            enable_gridstats=False
        )
        
        assert stats['network_name'] == 'test_net'
        assert stats['mse'] is not None
        assert stats['rmse'] is not None
        assert stats['mse'] >= 0
        assert stats['rmse'] >= 0
    
    def test_calculate_stats_single_output_pos(self, mock_rescaler):
        """Test with single integer output position."""
        yhat = np.random.rand(100, 3).astype(np.float32)
        gt = np.random.rand(100, 1).astype(np.float32)
        x = np.random.rand(100, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=gt,
            x=x,
            dependent_output_pos=0,  # Single integer
            nb_points_in_eval=100,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_net'},
            enable_gridstats=False
        )
        
        assert stats['mse'] is not None
        assert stats['rmse'] is not None
    
    def test_calculate_stats_empty_output(self, mock_rescaler):
        """Test with empty output selection."""
        yhat = np.random.rand(100, 3).astype(np.float32)
        x = np.random.rand(100, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=None,
            x=x,
            dependent_output_pos=[],  # Empty selection
            nb_points_in_eval=100,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_net'},
            enable_gridstats=False
        )
        
        assert 'error' in stats
        assert stats['latent_mean'] != stats['latent_mean']  # NaN check
    
    def test_calculate_stats_gt_fewer_columns(self, mock_rescaler):
        """Test with ground truth having fewer columns than predictions."""
        yhat = np.random.rand(100, 3).astype(np.float32)  # 3 total outputs
        gt = np.random.rand(100, 2).astype(np.float32)    # Only 2 dependent outputs
        x = np.random.rand(100, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=gt,
            x=x,
            dependent_output_pos=[1, 2],  # Last 2 columns are dependent
            nb_points_in_eval=100,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_net'},
            enable_gridstats=False
        )
        
        assert stats['mse'] is not None
        assert stats['rmse'] is not None
        assert stats['network_name'] == 'test_net'


class TestBroadcastingFix:
    """Test the original broadcasting fix that motivated the refactor."""
    
    def test_single_output_broadcasting(self):
        """Test that single output positions don't cause broadcasting errors."""
        mock_rescaler = Mock()
        mock_rescaler.fwd = Mock(side_effect=lambda x: x)
        
        # This scenario caused the original error
        yhat = np.random.rand(8192, 3).astype(np.float32)  # 3 outputs total
        gt = np.random.rand(8192, 1).astype(np.float32)    # Only 1 dependent output
        x = np.random.rand(8192, 2).astype(np.float32)
        
        # When output_pos is a single integer (like in get_reordered_protein_names)
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=gt,
            x=x,
            dependent_output_pos=2,  # Single integer (was causing issues)
            nb_points_in_eval=8192,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_broadcasting'},
            enable_gridstats=False
        )
        
        # Should not raise broadcasting error
        assert stats['mse'] is not None
        assert stats['rmse'] is not None
        assert stats['network_name'] == 'test_broadcasting'
    
    def test_multiple_output_broadcasting(self):
        """Test multiple output positions work correctly."""
        mock_rescaler = Mock()
        mock_rescaler.fwd = Mock(side_effect=lambda x: x)
        
        yhat = np.random.rand(1000, 5).astype(np.float32)  # 5 outputs total
        gt = np.random.rand(1000, 3).astype(np.float32)    # 3 dependent outputs
        x = np.random.rand(1000, 2).astype(np.float32)
        
        stats = _calculate_single_network_stats(
            network_idx=0,
            yhat=yhat,
            gt=gt,
            x=x,
            dependent_output_pos=[1, 2, 4],  # Multiple positions
            nb_points_in_eval=1000,
            rescaler=mock_rescaler,
            gridstats_params={},
            network_info={'network_name': 'test_multi_output'},
            enable_gridstats=False
        )
        
        assert stats['mse'] is not None
        assert stats['rmse'] is not None


if __name__ == "__main__":
    pytest.main([__file__])