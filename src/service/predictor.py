"""The thing behind POST /evaluate.  OWNER: Pranoy.

Two implementations, same interface:
    StubPredictor  - plausible numbers, no model, no GPU. BUILD THIS FIRST
                     (Oct 13) so Paramveer is not blocked on Avneet.
    TorchPredictor - loads one of Avneet's checkpoints. Swap it in Oct 15.

Make the stub's uncertainty RISE toward the edges of the design box, or the
uncertainty gate has nothing to fire on and cannot be tested.
"""

from __future__ import annotations

from ..contracts import Design, Evaluation


class BasePredictor:
    name = "base"

    def evaluate(self, design: Design) -> Evaluation:
        raise NotImplementedError


class StubPredictor(BasePredictor):
    name = "stub"

    def evaluate(self, design: Design) -> Evaluation:
        raise NotImplementedError("TODO Pranoy: Oct 13")


class TorchPredictor(BasePredictor):
    name = "torch"

    def __init__(self, checkpoint: str, device: str = "cpu"):
        raise NotImplementedError("TODO Pranoy: Oct 15")

    def evaluate(self, design: Design) -> Evaluation:
        raise NotImplementedError


def load_predictor(checkpoint: str | None = None) -> BasePredictor:
    """Torch predictor when a checkpoint is given, else the stub."""
    raise NotImplementedError("TODO Pranoy")
