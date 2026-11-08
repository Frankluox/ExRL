# Code structure

Start with the learning objective in [`exrl/algorithms/dp_q.py`](../exrl/algorithms/dp_q.py). The critic receives the state and predicted action chunk and returns a value for each execution length. The replay stores the executed length so the TD target uses the corresponding accumulated reward and discount. CQL compares values across possible lengths with the length observed in replay.

| Location | Purpose |
| --- | --- |
| `train.py`, `evaluate.py`, `pretrain.py` | Command-line entry points |
| `exrl/config.py` | YAML defaults, overrides, and experiment validation |
| `exrl/algorithms/` | Critic, actor, replay, CQL, and TD updates |
| `exrl/common/networks.py` | Shared actor and critic networks |
| `exrl/runners/dp_q/` | DP ExRL-Q collection loop and online BC |
| `exrl/runners/act_q/` | ACT ExRL-Q, online BC, and demonstration BC |
| `exrl/runners/dp_sac/`, `act_sac/` | ExRL-SAC collection loops |
| `exrl/runners/dp_demo/` | Demonstration segments as execution-length replay |
| `exrl/runners/act_pretrain/` | ACT pretraining and dataset loaders |
| `vendor/` | Bundled upstream dependencies |

Within each runner, `train.py` coordinates collection and updates, `environment.py` connects the task to chunk execution, `policy_loader.py` loads the base policy, and `options.py` defines advanced settings. Public experiment configurations select the settings needed for the paper experiments. Algorithm-specific replay formats and update rules live in their respective modules.

Online BC uses a separate successful-trajectory buffer in `online_bc.py`. Sampling respects episode boundaries. The base-policy loss and the initial-policy constraint are implemented in `policy_adapter.py` and `diffusion.py`. ACT's demonstration mixture and fractional update schedule are in `act_offline_bc.py` and `bc_update_schedule.py`.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

The CPU tests cover configuration resolution, variable-length TD targets, CQL, and BC replay. The integration tests additionally exercise data preparation, tiny DP/ACT pretraining, and checkpoint reloads. They use synthetic data and do not reproduce the paper's learning curves.
