"""PA-RL for ACT, using the shared action-optimization training loop."""

from baselines.parl.dp_train import TrainPARLDQLDiffusionAgent


class TrainPARLDQLACTAgent(TrainPARLDQLDiffusionAgent):
    """Select the ACT model and its native distillation objective through config."""
