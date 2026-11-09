"""Local TensorBoard logging, with optional Weights & Biases logging."""

from pathlib import Path
import numbers
import numpy as np


class WandBLogger:
    def __init__(
        self,
        wandb_logging,
        variant,
        project,
        experiment_id,
        output_dir=None,
        group_name="",
        team=None,
    ):
        from tensorboardX import SummaryWriter

        directory = Path(getattr(variant, "outputdir", None) or output_dir or "outputs")
        directory.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(directory / "scalars"))
        self.run = None
        if wandb_logging:
            import wandb

            self.run = wandb.init(
                project=project,
                name=experiment_id,
                config=dict(variant),
                dir=str(directory),
                group=group_name or None,
                entity=team,
            )

    def log(self, values, step=None, tb_step=None, **kwargs):
        local_step = tb_step if tb_step is not None else step
        if local_step is not None:
            for key, value in values.items():
                if isinstance(value, numbers.Number) or np.ndim(value) == 0:
                    try:
                        self.writer.add_scalar(key, float(value), int(local_step))
                    except (TypeError, ValueError):
                        pass
            self.writer.flush()
        if self.run is not None:
            self.run.log(values, step=step, **kwargs)

    def log_histogram(self, name, values, step, tb_step=None):
        self.writer.add_histogram(
            name, np.asarray(values), tb_step if tb_step is not None else step
        )

    def close(self):
        self.writer.close()
        if self.run is not None:
            self.run.finish()
