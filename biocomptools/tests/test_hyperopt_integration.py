"""Integration tests for hyperparameter optimization.

These tests verify that hyperparameters are correctly propagated from
Optuna trials through to the training/design execution.

Critical verification points:
1. HyperparamSpec.suggest() generates correct value types
2. Values reach the loss function via params tree (for design hyperopt)
3. Values reach training config (for training hyperopt)
"""

import pytest
import jax.numpy as jnp
import optuna

from biocomptools.hyperopt.base import (
    HyperparamSpec,
    verify_hyperparam_propagation,
    expand_schedule_hyperparams,
    get_schedule_param_names,
)
from biocomp.parameters import ParameterTree
from biocomp.designloss import (
    normalize_schedule_spec,
    init_schedule_params,
    _get_schedule_value,
    HYPEROPT_SCHEDULE_NAMESPACE,
)


class TestHyperparamSpec:
    def test_suggest_float(self):
        spec = HyperparamSpec(name='test_param', type='float', low=0.0, high=1.0)
        study = optuna.create_study()
        trial = study.ask()
        value = spec.suggest(trial)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_suggest_log_float(self):
        spec = HyperparamSpec(name='test_param', type='log_float', low=1e-6, high=1.0)
        study = optuna.create_study()
        trial = study.ask()
        value = spec.suggest(trial)
        assert isinstance(value, float)
        assert 1e-6 <= value <= 1.0

    def test_suggest_int(self):
        spec = HyperparamSpec(name='test_param', type='int', low=1, high=10)
        study = optuna.create_study()
        trial = study.ask()
        value = spec.suggest(trial)
        assert isinstance(value, int)
        assert 1 <= value <= 10

    def test_suggest_categorical(self):
        spec = HyperparamSpec(name='test_param', type='categorical', choices=['a', 'b', 'c'])
        study = optuna.create_study()
        trial = study.ask()
        value = spec.suggest(trial)
        assert value in ['a', 'b', 'c']

    def test_missing_low_high_raises(self):
        spec = HyperparamSpec(name='test_param', type='float')
        study = optuna.create_study()
        trial = study.ask()
        with pytest.raises(AssertionError):
            spec.suggest(trial)

    def test_different_trials_can_get_different_values(self):
        spec = HyperparamSpec(name='test_param', type='float', low=0.0, high=1.0)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
        values = set()
        for _ in range(20):
            trial = study.ask()
            value = spec.suggest(trial)
            values.add(value)
            study.tell(trial, 0.0)
        assert len(values) > 1, "Random sampler should produce different values"


class TestScheduleParamPropagation:
    def test_schedule_params_affect_value(self):
        """Verify that changing schedule params actually changes the output value."""
        params1 = ParameterTree()
        for path, value in init_schedule_params({'test_weight': 0.1}).items():
            params1[path] = value

        params2 = ParameterTree()
        for path, value in init_schedule_params({'test_weight': 2.0}).items():
            params2[path] = value

        total_steps = 100
        step = 50

        val1 = _get_schedule_value(
            params1, step, total_steps, 'test_weight', 999.0, HYPEROPT_SCHEDULE_NAMESPACE
        )
        val2 = _get_schedule_value(
            params2, step, total_steps, 'test_weight', 999.0, HYPEROPT_SCHEDULE_NAMESPACE
        )

        assert abs(float(val1) - float(val2)) > 1.0, (
            f"Different schedule params should produce different values: {val1} vs {val2}"
        )

    def test_schedule_injection_overwrites_correctly(self):
        """Verify that injecting new schedule values overwrites existing ones."""
        params = ParameterTree()
        initial_specs = init_schedule_params({'w_sinkhorn': 0.5})
        for path, value in initial_specs.items():
            params[path] = value

        initial_val = float(
            _get_schedule_value(params, 0, 100, 'w_sinkhorn', 999.0, HYPEROPT_SCHEDULE_NAMESPACE)
        )
        assert initial_val == pytest.approx(0.5)

        updated_specs = init_schedule_params({'w_sinkhorn': 3.0})
        for path, value in updated_specs.items():
            params[path] = value

        updated_val = float(
            _get_schedule_value(params, 0, 100, 'w_sinkhorn', 999.0, HYPEROPT_SCHEDULE_NAMESPACE)
        )
        assert updated_val == pytest.approx(3.0)

    def test_three_phase_schedule_evolves_over_steps(self):
        """Verify that schedule values change over the optimization steps."""
        specs = {
            'lambda_l0': {
                'phase1_frac': 0.3,
                'phase2_frac': 0.7,
                'phase1_value': 0.0,
                'phase2_end_value': 0.05,
                'phase3_end_value': 0.1,
            }
        }
        params = ParameterTree()
        for path, value in init_schedule_params(specs).items():
            params[path] = value

        total_steps = 100
        val_early = float(
            _get_schedule_value(params, 10, total_steps, 'lambda_l0', 0.0, HYPEROPT_SCHEDULE_NAMESPACE)
        )
        val_mid = float(
            _get_schedule_value(params, 50, total_steps, 'lambda_l0', 0.0, HYPEROPT_SCHEDULE_NAMESPACE)
        )
        val_late = float(
            _get_schedule_value(params, 90, total_steps, 'lambda_l0', 0.0, HYPEROPT_SCHEDULE_NAMESPACE)
        )

        assert val_early < val_mid < val_late, (
            f"Schedule should increase over time: early={val_early}, mid={val_mid}, late={val_late}"
        )


class TestVerifyHyperparamPropagation:
    def test_matching_params_returns_empty(self):
        expected = {'w_sinkhorn': 0.5, 'w_lncc': 0.3}
        actual = {'w_sinkhorn': 0.5, 'w_lncc': 0.3}
        discrepancies = verify_hyperparam_propagation({}, actual, expected)
        assert discrepancies == []

    def test_mismatched_value_detected(self):
        expected = {'w_sinkhorn': 0.5}
        actual = {'w_sinkhorn': 0.8}
        discrepancies = verify_hyperparam_propagation({}, actual, expected)
        assert len(discrepancies) == 1
        assert 'w_sinkhorn' in discrepancies[0]

    def test_missing_param_detected(self):
        expected = {'w_sinkhorn': 0.5}
        actual = {}
        discrepancies = verify_hyperparam_propagation({}, actual, expected)
        assert len(discrepancies) == 1
        assert 'not found' in discrepancies[0]

    def test_seed_is_skipped(self):
        expected = {'w_sinkhorn': 0.5, 'seed': 42}
        actual = {'w_sinkhorn': 0.5}
        discrepancies = verify_hyperparam_propagation({}, actual, expected)
        assert discrepancies == []


class TestDesignScheduleIntegration:
    def test_multiple_schedule_params_independent(self):
        """Verify that multiple schedules maintain independent values."""
        params = ParameterTree()
        specs = init_schedule_params({
            'w_sinkhorn': 1.0,
            'w_lncc': 0.1,
            'w_mse': 2.0,
        })
        for path, value in specs.items():
            params[path] = value

        total_steps = 100
        step = 50

        v_sink = float(_get_schedule_value(
            params, step, total_steps, 'w_sinkhorn', 999.0, HYPEROPT_SCHEDULE_NAMESPACE
        ))
        v_lncc = float(_get_schedule_value(
            params, step, total_steps, 'w_lncc', 999.0, HYPEROPT_SCHEDULE_NAMESPACE
        ))
        v_mse = float(_get_schedule_value(
            params, step, total_steps, 'w_mse', 999.0, HYPEROPT_SCHEDULE_NAMESPACE
        ))

        assert v_sink == pytest.approx(1.0)
        assert v_lncc == pytest.approx(0.1)
        assert v_mse == pytest.approx(2.0)

    def test_schedule_namespace_isolation(self):
        """Verify that schedule params are in correct namespace."""
        specs = init_schedule_params({'test_weight': 0.5})

        for path in specs.keys():
            assert path.startswith(HYPEROPT_SCHEDULE_NAMESPACE), (
                f"Schedule param {path} should be in {HYPEROPT_SCHEDULE_NAMESPACE} namespace"
            )


class TestExpandScheduleHyperparams:
    """Tests for expand_schedule_hyperparams helper function."""

    def test_constant_schedule_expands_to_three_phases(self):
        """Constant value should expand to all 3 phases with same value."""
        hp = {'w_sinkhorn': 0.5, 'phase1_frac': 0.3, 'phase2_frac': 0.7}
        expanded = expand_schedule_hyperparams(hp)

        assert expanded['w_sinkhorn_phase1_value'] == 0.5
        assert expanded['w_sinkhorn_phase2_end_value'] == 0.5
        assert expanded['w_sinkhorn_phase3_end_value'] == 0.5

    def test_linear_schedule_computes_phase2(self):
        """Linear schedule (phase1 + phase3) should compute phase2 via interpolation."""
        hp = {
            'w_lncc_phase1_value': 0.0,
            'w_lncc_phase3_end_value': 1.0,
            'phase1_frac': 0.25,
            'phase2_frac': 0.75,
        }
        expanded = expand_schedule_hyperparams(hp)

        # At phase2_frac=0.75, with linear interp from phase1=0.25 to end=1.0:
        # t = (0.75 - 0.25) / (1.0 - 0.25) = 0.5 / 0.75 = 0.6667
        # phase2 = 0.0 + (1.0 - 0.0) * 0.6667 = 0.6667
        assert expanded['w_lncc_phase2_end_value'] == pytest.approx(0.6667, rel=1e-3)

    def test_full_3phase_unchanged(self):
        """Full 3-phase spec should pass through unchanged."""
        hp = {
            'lambda_l0_phase1_value': 0.0,
            'lambda_l0_phase2_end_value': 0.05,
            'lambda_l0_phase3_end_value': 0.1,
            'phase1_frac': 0.3,
            'phase2_frac': 0.7,
        }
        expanded = expand_schedule_hyperparams(hp)

        assert expanded['lambda_l0_phase1_value'] == 0.0
        assert expanded['lambda_l0_phase2_end_value'] == 0.05
        assert expanded['lambda_l0_phase3_end_value'] == 0.1

    def test_incomplete_schedule_raises(self):
        """Incomplete schedule (e.g., only phase2) should raise ValueError."""
        hp = {
            'w_bad_phase2_end_value': 0.5,  # missing phase1 and phase3
            'phase1_frac': 0.3,
            'phase2_frac': 0.7,
        }
        with pytest.raises(ValueError, match="incomplete spec"):
            expand_schedule_hyperparams(hp)

    def test_invalid_phase_fractions_raises(self):
        """phase1_frac >= phase2_frac should raise AssertionError."""
        hp = {'w_test': 0.5, 'phase1_frac': 0.8, 'phase2_frac': 0.3}
        with pytest.raises(AssertionError, match="Invalid phase fractions"):
            expand_schedule_hyperparams(hp)

    def test_multiple_schedules_mixed_types(self):
        """Multiple schedules with different types should all be handled."""
        hp = {
            'w_sinkhorn': 0.5,  # constant
            'w_lncc_phase1_value': 0.0,  # linear
            'w_lncc_phase3_end_value': 1.0,
            'lambda_l0_phase1_value': 0.0,  # 3-phase
            'lambda_l0_phase2_end_value': 0.05,
            'lambda_l0_phase3_end_value': 0.1,
            'phase1_frac': 0.3,
            'phase2_frac': 0.7,
        }
        expanded = expand_schedule_hyperparams(hp)

        # Check constant
        assert expanded['w_sinkhorn_phase1_value'] == 0.5
        assert expanded['w_sinkhorn_phase2_end_value'] == 0.5
        assert expanded['w_sinkhorn_phase3_end_value'] == 0.5

        # Check linear has computed phase2
        assert 'w_lncc_phase2_end_value' in expanded
        assert 0 < expanded['w_lncc_phase2_end_value'] < 1

        # Check 3-phase unchanged
        assert expanded['lambda_l0_phase2_end_value'] == 0.05


class TestGetScheduleParamNames:
    """Tests for get_schedule_param_names helper function."""

    def test_identifies_constant_schedules(self):
        specs = [HyperparamSpec(name='w_sinkhorn', type='float', low=0.0, high=1.0)]
        result = get_schedule_param_names(specs)
        assert 'w_sinkhorn' in result
        assert '' in result['w_sinkhorn']  # empty string = base name

    def test_identifies_linear_schedules(self):
        specs = [
            HyperparamSpec(name='w_lncc_phase1_value', type='float', low=0.0, high=1.0),
            HyperparamSpec(name='w_lncc_phase3_end_value', type='float', low=0.0, high=1.0),
        ]
        result = get_schedule_param_names(specs)
        assert 'w_lncc' in result
        assert '_phase1_value' in result['w_lncc']
        assert '_phase3_end_value' in result['w_lncc']
        assert '_phase2_end_value' not in result['w_lncc']

    def test_identifies_full_3phase_schedules(self):
        specs = [
            HyperparamSpec(name='lambda_l0_phase1_value', type='float', low=0.0, high=0.01),
            HyperparamSpec(name='lambda_l0_phase2_end_value', type='float', low=0.0, high=0.1),
            HyperparamSpec(name='lambda_l0_phase3_end_value', type='float', low=0.0, high=0.5),
        ]
        result = get_schedule_param_names(specs)
        assert 'lambda_l0' in result
        assert len(result['lambda_l0']) == 3

    def test_ignores_phase_fractions_and_seed(self):
        specs = [
            HyperparamSpec(name='phase1_frac', type='float', low=0.1, high=0.4),
            HyperparamSpec(name='phase2_frac', type='float', low=0.5, high=0.9),
            HyperparamSpec(name='w_test', type='float', low=0.0, high=1.0),
        ]
        result = get_schedule_param_names(specs)
        assert 'phase1_frac' not in result
        assert 'phase2_frac' not in result
        assert 'w_test' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
