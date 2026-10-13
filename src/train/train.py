"""Stage 1 trainer.  OWNER: Avneet.  GATE: real R^2 posted by Oct 15.

    python -m src.train.train --config configs/baseline.yaml

If you cannot hit the gate by the 15th, switch to AirfRANS. Do not keep pushing.

Save checkpoints Pranoy's TorchPredictor can load:
    {"state_dict", "config", "norm_stats", "version", "val_metrics"}

Normalise targets so the loss is well conditioned; undo it at inference.
"""

from __future__ import annotations


def r2_score(y_true, y_pred) -> float:
    raise NotImplementedError("TODO Avneet")


def evaluate(model, loader, device, norm) -> dict:
    raise NotImplementedError("TODO Avneet")


def main() -> None:
    """Train, early-stop on val, save best.pt, print the test R2.
    Post that number in the group chat -- it is the Oct 15 gate."""
    raise NotImplementedError("TODO Avneet: Oct 15")


if __name__ == "__main__":
    main()
