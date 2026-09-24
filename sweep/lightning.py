"""Lightning pieces that let checkpoints be selected on the fixed ruler, not ``val_loss``."""

import numpy as np
import pytorch_lightning as L
import torch

from model import LightningWBoson
from sweep import metrics, selection

# Logged in place of a non-finite composite, so checkpoint selection still works.
UNUSABLE_COMPOSITE = 1.0e6


class SweepLightningWBoson(LightningWBoson):
    """Production model with reusable validation outputs for fixed-ruler metrics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_validation_outputs = False
        self._validation_outputs = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, aux = self(x, return_aux=True)
        total, losses = self._compute_losses(x, y, y_pred, aux)
        self._log_losses("val_", losses, total)
        if self.collect_validation_outputs:
            self._validation_outputs.append(
                (x.detach().cpu(), y.detach().cpu(), y_pred.detach().cpu())
            )
        return total

    def take_validation_outputs(self):
        """Hand over the stashed validation batches and forget them."""
        stashed = self._validation_outputs
        self._validation_outputs = []
        if not stashed:
            return None
        return tuple(torch.cat(column) for column in zip(*stashed))


class FidelityMetrics(L.Callback):
    """Score the fixed ruler on the predictions ``validation_step`` already made."""

    SUMMARY_PREFIX = "fid/"
    COMPOSITE_METRIC = "fid/composite"
    SELECTION_METRIC = "fid/selection_score"

    def __init__(self, weights, thresholds=None, every_n_epochs=1):
        super().__init__()
        self.weights = dict(weights)
        self.thresholds = thresholds
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.last_composite = None

    def _scheduled(self, trainer):
        return not trainer.sanity_checking and trainer.current_epoch % self.every_n_epochs == 0

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.collect_validation_outputs = self._scheduled(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        collected = pl_module.take_validation_outputs()
        pl_module.collect_validation_outputs = False
        if collected is None:
            return

        x, y_true, y_pred = collected
        report = metrics.evaluate(x, y_pred, y_true)
        score = metrics.composite(report["objectives"], self.weights)
        summary = self._summary(report, score)
        self.last_composite = summary[self.COMPOSITE_METRIC]

        for name, value in summary.items():
            pl_module.log(name, float(value), on_step=False, on_epoch=True, prog_bar=False)

    def _summary(self, report, score):
        """The handful of numbers worth a column in the per-epoch history."""
        usable_score = score if np.isfinite(score) else UNUSABLE_COMPOSITE
        selection_score = usable_score
        if self.thresholds is not None:
            violations = selection.constraint_violations(report, self.thresholds)
            report["constraint_violations"] = violations
            report["feasible"] = all(value <= 0.0 for value in violations.values())
            positive = [value for value in violations.values() if value > 0.0]
            if any(not np.isfinite(value) for value in positive):
                selection_score = UNUSABLE_COMPOSITE
            elif positive:
                selection_score = usable_score + 1000.0 * (1.0 + sum(positive))
        summary = {
            self.COMPOSITE_METRIC: usable_score,
            self.SELECTION_METRIC: selection_score,
        }
        for name, value in report["objectives"].items():
            summary[f"{self.SUMMARY_PREFIX}{name}"] = value
        for group in ("single", "pair"):
            for name, entry in report["distributions"][group].items():
                summary[f"{self.SUMMARY_PREFIX}tv/{name}"] = entry["tv"]
        for name, entry in report["distributions"]["joint"].items():
            summary[f"{self.SUMMARY_PREFIX}tv2d/{name}"] = entry["tv"]
        summary[f"{self.SUMMARY_PREFIX}max_bias_over_resolution"] = max(
            entry["bias_over_resolution"] for entry in report["bias_components"].values()
        )
        return summary
