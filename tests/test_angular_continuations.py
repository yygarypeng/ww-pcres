import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from model import LightningWBoson
from scripts.run_angular_continuations import (
    build_continuation_specs,
    build_train_command,
    run_sequentially,
)
from train.train import ContinuationTreatmentCallback, build_training_callbacks


class ContinuationSpecificationTest(unittest.TestCase):
    def setUp(self):
        self.base = {
            "parameters": {
                "learning_rate": 5.0e-4,
                "epochs": 1024,
                "num_workers": 8,
                "persistent_workers": True,
                "loss_weights": {"huber": 50.0, "angular_mmd": 2000.0},
            },
            "mmd": {
                "local": False,
                "angular": {
                    "kernel": "imq",
                    "bandwidth_multipliers": [0.05, 0.5, 5.0],
                },
            },
            "paths": {"saved_path": "outputs/original", "data_path": "local-data.h5"},
        }
        self.bandwidths = [0.5, 1.0, 2.0, 4.0]

    def test_a_has_no_treatment_override(self):
        specs = build_continuation_specs(
            self.base,
            self.bandwidths,
            output_root=Path("outputs/angular_diagnosis/continuations"),
        )

        run_a = specs[0]
        self.assertEqual(run_a.name, "A")
        self.assertNotIn("continuation_treatment", run_a.config)
        self.assertEqual(run_a.config["parameters"]["learning_rate"], 5.0e-4)

    def test_all_runs_share_control_settings_and_have_isolated_destinations(self):
        specs = build_continuation_specs(self.base, self.bandwidths, output_root=Path("runs"))

        self.assertEqual([spec.name for spec in specs], ["A", "B", "C", "D"])
        self.assertEqual(len({spec.config["paths"]["saved_path"] for spec in specs}), 4)
        for spec in specs:
            with self.subTest(run=spec.name):
                params = spec.config["parameters"]
                self.assertEqual(params["epochs"], 137)
                self.assertTrue(params["save_every_epoch"])
                self.assertTrue(params["disable_early_stopping"])
                self.assertEqual(params["num_workers"], 0)
                self.assertFalse(params["persistent_workers"])

    def test_treatments_change_only_the_declared_variable(self):
        specs = {
            spec.name: spec
            for spec in build_continuation_specs(self.base, self.bandwidths, output_root=Path("runs"))
        }

        self.assertEqual(specs["B"].config["continuation_treatment"], {"learning_rate": 5.0e-5})
        self.assertEqual(
            specs["C"].config["continuation_treatment"],
            {"angular_mmd_weight": 0.0},
        )
        self.assertEqual(
            specs["D"].config["continuation_treatment"],
            {
                "angular_mmd_weight": 2000.0,
                "angular_mmd_estimator": "v",
                "angular_mmd_feature_bandwidths": self.bandwidths,
            },
        )
        self.assertEqual(self.base["parameters"]["epochs"], 1024)

    def test_train_command_uses_full_checkpoint_resume(self):
        command = build_train_command(
            config_path=Path("run-configs/A.yaml"),
            checkpoint=Path("baseline/checkpoints/epoch=134.ckpt"),
            gpu=1,
        )

        self.assertEqual(
            command,
            [
                sys.executable,
                "train/train.py",
                "--config",
                "run-configs/A.yaml",
                "--resume-from",
                "baseline/checkpoints/epoch=134.ckpt",
                "--gpu",
                "1",
            ],
        )

    def test_runner_is_sequential_and_stops_after_first_failure(self):
        commands = [["train", "A"], ["train", "B"], ["train", "C"]]
        run = Mock(
            side_effect=[
                subprocess.CompletedProcess(commands[0], 0),
                subprocess.CalledProcessError(7, commands[1]),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "B.*exit code 7"):
            run_sequentially(zip(("A", "B", "C"), commands), run_command=run)

        self.assertEqual([call.args[0] for call in run.call_args_list], commands[:2])


class ContinuationRestoreTreatmentTest(unittest.TestCase):
    def test_learning_rate_is_replaced_after_optimizer_restore(self):
        callback = ContinuationTreatmentCallback({"learning_rate": 5.0e-5})
        optimizer = SimpleNamespace(param_groups=[{"lr": 5.0e-4}, {"lr": 2.0e-4}])
        trainer = SimpleNamespace(optimizers=[optimizer])
        model = SimpleNamespace()

        callback.on_train_start(trainer, model)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [5.0e-5, 5.0e-5])

    def test_model_treatment_is_reapplied_after_checkpoint_restore(self):
        treatment = {
            "angular_mmd_weight": 0.0,
            "angular_mmd_estimator": "v",
            "angular_mmd_feature_bandwidths": [0.5, 1.0, 2.0, 4.0],
        }
        callback = ContinuationTreatmentCallback(treatment)
        model = SimpleNamespace(
            loss_weights={"angular_mmd": 2000.0},
            angular_mmd_estimator="u",
            angular_mmd_feature_bandwidths=None,
        )

        callback.on_train_start(SimpleNamespace(optimizers=[]), model)

        self.assertEqual(model.loss_weights["angular_mmd"], 0.0)
        self.assertEqual(model.angular_mmd_estimator, "v")
        self.assertEqual(model.angular_mmd_feature_bandwidths, [0.5, 1.0, 2.0, 4.0])

    def test_control_callbacks_contain_no_treatment_callback(self):
        callbacks = build_training_callbacks(
            {"save_every_epoch": True, "disable_early_stopping": True},
            continuation_treatment=None,
        )

        self.assertFalse(any(isinstance(item, ContinuationTreatmentCallback) for item in callbacks))

    def test_early_stopping_is_disabled_without_changing_checkpoint_or_rng_callbacks(self):
        callbacks = build_training_callbacks(
            {"save_every_epoch": True, "disable_early_stopping": True},
            continuation_treatment={"learning_rate": 5.0e-5},
        )

        self.assertEqual(callbacks[0].save_top_k, -1)
        self.assertFalse(any(item.__class__.__name__ == "DeferredEarlyStopping" for item in callbacks))
        self.assertTrue(any(item.__class__.__name__ == "RNGStateCallback" for item in callbacks))
        self.assertTrue(any(isinstance(item, ContinuationTreatmentCallback) for item in callbacks))


class AngularMmdContinuationRoutingTest(unittest.TestCase):
    def test_fixed_v_statistic_is_routed_only_to_angular_mmd(self):
        model = LightningWBoson(
            input_dim=21,
            d_model=8,
            num_heads=2,
            std_mean_train=np.zeros(21, dtype=np.float32),
            std_scale_train=np.ones(21, dtype=np.float32),
            mmd_cond_mean_train=np.zeros(3, dtype=np.float32),
            mmd_cond_scale_train=np.ones(3, dtype=np.float32),
            angular_mmd_estimator="v",
            angular_mmd_feature_bandwidths=[0.5, 1.0, 2.0, 4.0],
            attention_blocks=1,
            attention_dropout=0.0,
            decoder_dropout=0.0,
        )

        angular = model._mmd_kwargs("angular")
        mass = model._mmd_kwargs("mass")

        self.assertEqual(angular["estimator"], "v")
        self.assertEqual(angular["feature_bandwidths"], [0.5, 1.0, 2.0, 4.0])
        self.assertNotIn("estimator", mass)
        self.assertNotIn("feature_bandwidths", mass)


if __name__ == "__main__":
    unittest.main()
