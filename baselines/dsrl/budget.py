"""Optional primitive-step budget with full final checkpoint and terminal eval."""

import json
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class PrimitiveEnvBudgetCallback(BaseCallback):
    def __init__(self, target, latest_callback, logging_callback, status_path):
        super().__init__()
        self.target = int(target)
        if self.target <= 0:
            raise ValueError("A positive primitive env-step budget is required")
        self.latest = latest_callback
        self.logging = logging_callback
        self.status_path = Path(status_path)
        self.reached = False

    def _on_step(self):
        step = int(self.model.env_steps_exact)
        if step >= self.target:
            # Finish storing the current transition rather than returning False
            # before SB3's replay insertion; learn exits after this train_freq.
            self.model._total_timesteps = int(self.model.num_timesteps)
            if not self.reached:
                print(
                    "[primitive-budget] reached env_step=%d target=%d" % (step, self.target),
                    flush=True,
                )
            self.reached = True
        return True

    def _on_training_end(self):
        step = int(self.model.env_steps_exact)
        if step < self.target:
            raise RuntimeError(
                "Training ended before primitive budget: %d < %d" % (step, self.target)
            )
        self.latest.pending_save = True
        self.latest.next_retry_monotonic = 0.0
        self.latest._save_if_due()
        if self.latest.pending_save:
            raise RuntimeError("Final full-state checkpoint was not saved")
        proc = self.logging.online_eval_process
        if proc is not None:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("Previous evaluator failed: %s" % proc.exitcode)
        terminal = self.logging.run_online_eval(self.model, env_step=step)
        if terminal is not None:
            terminal.join()
            if terminal.exitcode != 0:
                raise RuntimeError("Terminal evaluator failed: %s" % terminal.exitcode)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(
            json.dumps(
                {
                    "state": "complete",
                    "env_steps": step,
                    "target_env_steps": self.target,
                    "terminal_eval_episodes": self.logging.online_eval_total_episodes,
                },
                indent=2,
            )
        )
