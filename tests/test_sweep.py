import numpy as np
import pytest
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from physics.torchBoost import _mock_inputs
from sweep import metrics, runner, space
from sweep.lightning import FidelityMetrics, SweepLightningWBoson


def test_full_epoch_callbacks_keep_checkpointing_and_disable_early_stopping():
    fidelity = FidelityMetrics(runner.DEFAULT_WEIGHTS)
    callbacks = runner._callbacks({"parameters": {}}, fidelity, [], full_epochs=True)
    assert not any(isinstance(callback, EarlyStopping) for callback in callbacks)
    assert isinstance(callbacks[-1], ModelCheckpoint)
    assert callbacks[0] is fidelity


class FakeTrial:
    """The slice of the optuna trial API that sweep.space.suggest uses."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def suggest_categorical(self, name, choices):
        self.asked.append(name)
        return self.answers.get(name, choices[0])

    def suggest_int(self, name, low, high, log=False):
        self.asked.append(name)
        return int(self.answers.get(name, low))

    def suggest_float(self, name, low, high, log=False):
        self.asked.append(name)
        return float(self.answers.get(name, low))


def write_space(tmp_path, body):
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def test_a_gated_axis_stays_absent_from_the_config_when_its_gate_is_off(tmp_path):
    # A log axis cannot reach zero, so "penalty off" is only expressible as a gate.
    path = write_space(
        tmp_path,
        {
            "axes": {
                "sweep.bias_penalty": {"type": "categorical", "choices": ["off", "on"]},
                "parameters.loss_weights.fourvec_bias": {
                    "type": "float",
                    "low": 0.001,
                    "high": 1000.0,
                    "log": True,
                    "when": {"sweep.bias_penalty": "on"},
                },
            }
        },
    )
    sweep_space = space.load_space(path)
    base = {"parameters": {"loss_weights": {"w_fourvec": 0.3}}}

    off = space.apply(base, space.suggest(sweep_space, FakeTrial({"sweep.bias_penalty": "off"})))
    on = space.apply(base, space.suggest(sweep_space, FakeTrial({"sweep.bias_penalty": "on"})))

    assert "fourvec_bias" not in off["parameters"]["loss_weights"]
    assert on["parameters"]["loss_weights"]["fourvec_bias"] == 0.001


def test_load_space_rejects_unknown_type(tmp_path):
    path = write_space(tmp_path, {"axes": {"parameters.epochs": {"type": "boolean"}}})
    with pytest.raises(ValueError, match="unsupported type"):
        space.load_space(path)


def test_load_space_rejects_empty_axes(tmp_path):
    path = write_space(tmp_path, {"axes": {}})
    with pytest.raises(ValueError, match="no axes"):
        space.load_space(path)


def test_load_space_rejects_forward_dependency(tmp_path):
    path = write_space(
        tmp_path,
        {
            "axes": {
                "parameters.huber_delta": {
                    "type": "float",
                    "low": 1.0,
                    "high": 2.0,
                    "when": {"parameters.fourvec_loss": "huber"},
                },
                "parameters.fourvec_loss": {
                    "type": "categorical",
                    "choices": ["l1", "huber"],
                },
            }
        },
    )
    with pytest.raises(ValueError, match="must come after its dependency"):
        space.load_space(path)


def test_suggest_skips_inactive_conditional_axis():
    sweep_space = {
        "axes": {
            "parameters.fourvec_loss": {
                "type": "categorical",
                "choices": ["l1", "huber"],
            },
            "parameters.huber_delta": {
                "type": "float",
                "low": 2.0,
                "high": 60.0,
                "when": {"parameters.fourvec_loss": "huber"},
            },
        }
    }
    chosen = space.suggest(sweep_space, FakeTrial({"parameters.fourvec_loss": "l1"}))
    assert chosen == {"parameters.fourvec_loss": "l1"}

    chosen = space.suggest(sweep_space, FakeTrial({"parameters.fourvec_loss": "huber"}))
    assert chosen == {"parameters.fourvec_loss": "huber", "parameters.huber_delta": 2.0}


def test_suggest_round_trips_list_choices():
    sweep_space = {
        "axes": {"mmd.angular.bandwidths": {"type": "categorical", "choices": [[1.0, 2.0]]}}
    }
    assert space.suggest(sweep_space, FakeTrial({})) == {"mmd.angular.bandwidths": [1.0, 2.0]}


def test_apply_writes_nested_paths_without_touching_the_base():
    base = {"parameters": {"loss_weights": {"angular_mmd": 20.0}}}
    applied = space.apply(base, {"parameters.loss_weights.angular_mmd": 60.0, "sweep.x": 1})
    assert applied["parameters"]["loss_weights"]["angular_mmd"] == 60.0
    assert applied["sweep"]["x"] == 1
    assert base["parameters"]["loss_weights"]["angular_mmd"] == 20.0


def test_baseline_point_reads_the_config_and_its_sweep_defaults():
    sweep_space = {
        "axes": {
            "parameters.loss_weights.angular_mmd": {
                "type": "categorical",
                "choices": [5.0, 20.0],
            },
            "parameters.fourvec_loss": {
                "type": "categorical",
                "choices": ["l1", "huber"],
            },
        }
    }
    base = {"parameters": {"loss_weights": {"angular_mmd": 20.0}}}
    assert space.baseline_point(sweep_space, base) == {
        "parameters.loss_weights.angular_mmd": 20.0,
        "parameters.fourvec_loss": "l1",
    }


def test_baseline_point_is_empty_when_the_config_is_outside_the_space():
    sweep_space = {
        "axes": {"parameters.batch_size": {"type": "categorical", "choices": [4096, 8192]}}
    }
    assert space.baseline_point(sweep_space, {"parameters": {"batch_size": 512}}) == {}


def test_total_variation_spans_identical_to_disjoint():
    edges = metrics.edges((0.0, 1.0), 10)
    sample = np.linspace(0.0, 1.0, 1000)
    assert metrics.total_variation(sample, sample, edges) == pytest.approx(0.0)
    assert metrics.total_variation(sample * 0.4, 0.6 + sample * 0.4, edges) == pytest.approx(1.0)


def test_chi2_per_ndf_vanishes_for_identical_histograms():
    edges = metrics.edges((0.0, 1.0), 10)
    sample = np.linspace(0.0, 1.0, 1000)
    assert metrics.chi2_per_ndf(sample, sample, edges) == pytest.approx(0.0)


def test_bias_report_recovers_a_known_offset():
    rng = np.random.default_rng(2330)
    truth = rng.normal(0.0, 10.0, size=(20000, 10))
    prediction = truth[:, :8] + 2.0
    components, directions, scores = metrics.bias_report(prediction, truth)

    assert components["w0_px"]["bias_gev"] == pytest.approx(2.0, abs=0.2)
    assert components["w0_px"]["resolution_gev"] == pytest.approx(0.0, abs=1e-6)
    # A pure offset has no residual spread, so the scaled bias is not finite.
    assert np.isnan(scores["bias"])
    # The same offset on both W bosons cancels in the split and doubles in the sum.
    assert directions["px_diff"]["bias_gev"] == pytest.approx(0.0, abs=0.2)
    assert directions["px_sum"]["bias_gev"] == pytest.approx(4.0, abs=0.4)


def test_bias_report_scales_by_the_residual_width():
    rng = np.random.default_rng(2330)
    truth = rng.normal(0.0, 10.0, size=(40000, 10))
    prediction = truth[:, :8] + rng.normal(1.0, 4.0, size=(40000, 8))
    components, _, scores = metrics.bias_report(prediction, truth)

    assert components["w1_energy"]["bias_gev"] == pytest.approx(1.0, abs=0.1)
    assert components["w1_energy"]["resolution_gev"] == pytest.approx(4.0, abs=0.1)
    assert components["w1_energy"]["bias_over_resolution"] == pytest.approx(0.25, abs=0.03)
    assert scores["bias_all"] == pytest.approx(0.25, abs=0.03)


def test_bias_score_separates_the_transverse_split_from_the_constrained_part():
    """An antisymmetric px offset moves bias and bias_split, not bias_constrained."""
    rng = np.random.default_rng(2330)
    truth = rng.normal(0.0, 10.0, size=(40000, 10))
    prediction = truth[:, :8] + rng.normal(0.0, 4.0, size=(40000, 8))
    _, _, clean = metrics.bias_report(prediction, truth)

    shifted = prediction.copy()
    shifted[:, 0] += 5.0  # w0_px
    shifted[:, 4] -= 5.0  # w1_px, so the MET-constrained sum is untouched
    _, directions, scores = metrics.bias_report(shifted, truth)

    assert directions["px_sum"]["bias_gev"] == pytest.approx(0.0, abs=0.2)
    assert directions["px_diff"]["bias_gev"] == pytest.approx(10.0, abs=0.3)
    # The goal is scored...
    assert scores["bias"] > clean["bias"] + 0.2
    assert scores["bias_split"] > clean["bias_split"] + 0.5
    # ...without contaminating the half that was already clean.
    assert scores["bias_constrained"] == pytest.approx(clean["bias_constrained"], abs=0.02)


def test_an_energy_offset_leaves_the_ranked_bias_untouched():
    """An energy offset shows up in ``bias_energy`` but not in the ranked ``bias``."""
    rng = np.random.default_rng(2330)
    truth = rng.normal(0.0, 10.0, size=(40000, 10))
    prediction = truth[:, :8] + rng.normal(0.0, 4.0, size=(40000, 8))
    _, _, clean = metrics.bias_report(prediction, truth)

    shifted = prediction.copy()
    shifted[:, 3] -= 6.0  # w0_energy
    shifted[:, 7] -= 6.0  # w1_energy
    _, directions, scores = metrics.bias_report(shifted, truth)

    assert directions["w0_energy"]["bias_gev"] == pytest.approx(-6.0, abs=0.2)
    assert scores["bias_energy"] > clean["bias_energy"] + 0.5
    # The ranked score and the momentum halves stay where they were.
    assert scores["bias"] == pytest.approx(clean["bias"], abs=0.02)
    assert scores["bias_split"] == pytest.approx(clean["bias_split"], abs=0.02)


def test_angular_observables_mask_truth_and_prediction_together():
    lep, wboson = _mock_inputs(batch=512)
    x = torch.cat([lep, torch.zeros(512, 8), torch.zeros(512, 2)], dim=-1)
    y_true = torch.cat([wboson, torch.full((512, 2), 80.4)], dim=-1)

    prediction, truth, kept = metrics.angular_observables(x, wboson, y_true)
    assert {len(values) for values in prediction.values()} == {int(kept.sum())}
    assert {len(values) for values in truth.values()} == {int(kept.sum())}


def test_a_perfect_prediction_scores_zero_on_every_distribution():
    lep, wboson = _mock_inputs(batch=2048)
    x = torch.cat([lep, torch.zeros(2048, 8), torch.zeros(2048, 2)], dim=-1)
    y_true = torch.cat([wboson, torch.full((2048, 2), 80.4)], dim=-1)

    report = metrics.evaluate(x, wboson, y_true)
    assert report["objectives"]["angular_1d"] == pytest.approx(0.0)
    assert report["objectives"]["interangular_1d"] == pytest.approx(0.0)
    assert report["joint_2d"] == pytest.approx(0.0)


def test_composite_weights_each_objective():
    objectives = {"bias": 0.1, "angular_1d": 0.2, "interangular_1d": 0.4}
    weights = {"bias": 1.0, "angular_1d": 2.0, "interangular_1d": 0.5}
    assert metrics.composite(objectives, weights) == pytest.approx(0.7)


def test_prepare_trial_dir_refuses_paths_outside_the_sweep_output_tree(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        runner.prepare_trial_dir(tmp_path / "trial")


def test_sweep_model_rejects_an_unknown_four_vector_loss():
    with pytest.raises(ValueError, match="fourvec_loss must be one of"):
        SweepLightningWBoson(
            input_dim=18,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(18),
            std_scale_train=np.ones(18),
            fourvec_loss="mae",
        )


def test_sweep_and_production_models_use_the_same_loss_config_keys():
    params = {
        "learning_rate": 4.0e-4,
        "weight_decay": 0.03,
        "loss_weights": {"w_fourvec": 0.3, "fourvec_bias": 17.0},
        "d_model": 8,
        "n_heads": 2,
        "attention_blocks": 1,
        "fourvec_loss": "huber",
        "huber_delta": 23.0,
    }
    cfg = {"parameters": params, "mmd": {}}
    standardization = (np.zeros(18, dtype=np.float32), np.ones(18, dtype=np.float32))

    swept = runner.build_model(cfg, 18, standardization)
    production = SweepLightningWBoson(
        input_dim=18,
        d_model=8,
        num_heads=2,
        std_mean_train=standardization[0],
        std_scale_train=standardization[1],
        attention_blocks=1,
        loss_weights=params["loss_weights"],
        fourvec_loss=params["fourvec_loss"],
        huber_delta=params["huber_delta"],
    )

    assert swept.fourvec_loss == production.fourvec_loss == "huber"
    assert swept.huber_delta == production.huber_delta == 23.0
    assert swept.loss_weights["fourvec_bias"] == production.loss_weights["fourvec_bias"] == 17.0


def _fidelity_report(angular_1d=0.1):
    return {
        "objectives": {"bias": 0.1, "angular_1d": angular_1d, "interangular_1d": 0.1},
        "bias_directions": {
            name: {"bias_gev": 0.1, "bias_over_resolution": 0.01}
            for name in metrics.SCORED_DIRECTIONS
        },
        "bias_components": {
            name: {"bias_over_resolution": 0.01} for name in metrics.W_COMPONENT_NAMES
        },
        "distributions": {"single": {}, "pair": {}, "joint": {}},
    }


def test_constrained_checkpoint_score_penalizes_only_violations():
    thresholds = {
        "angular_1d": 0.2,
        "interangular_1d": 0.2,
        "bias_gev": {name: 1.0 for name in metrics.SCORED_DIRECTIONS},
    }
    callback = FidelityMetrics(runner.DEFAULT_WEIGHTS, thresholds=thresholds)

    feasible = callback._summary(_fidelity_report(), 0.3)
    violated = callback._summary(_fidelity_report(angular_1d=0.3), 0.5)

    assert feasible[FidelityMetrics.SELECTION_METRIC] == pytest.approx(0.3)
    assert np.isfinite(violated[FidelityMetrics.SELECTION_METRIC])
    assert violated[FidelityMetrics.SELECTION_METRIC] > feasible[FidelityMetrics.SELECTION_METRIC]


def test_unconstrained_checkpoint_uses_the_composite():
    callback = FidelityMetrics(runner.DEFAULT_WEIGHTS)
    summary = callback._summary(_fidelity_report(), 0.3)
    assert summary[FidelityMetrics.SELECTION_METRIC] == pytest.approx(0.3)


def test_prepare_trial_dir_accepts_a_configured_output_root(tmp_path):
    trial_dir = tmp_path / "outputs_v2" / "trials" / "trial_0000"
    trial_dir.mkdir(parents=True)
    (trial_dir / "stale.txt").write_text("from a previous attempt")

    prepared = runner.prepare_trial_dir(trial_dir, output_root=tmp_path / "outputs_v2")

    assert prepared.exists()
    assert not (prepared / "stale.txt").exists()


def test_prepare_trial_dir_still_refuses_outside_the_configured_root(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(ValueError, match="refusing to clear"):
        runner.prepare_trial_dir(tmp_path / "elsewhere", output_root=tmp_path / "outputs_v2")
