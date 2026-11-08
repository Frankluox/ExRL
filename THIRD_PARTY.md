# Third-party software

The bundled dependencies retain their upstream copyright and license notices.

| Component | Location | License |
| --- | --- | --- |
| Robomimic | `vendor/robomimic` | MIT: `licenses/robomimic.LICENSE` |
| DPPO / DSRL utilities | `vendor/dsrl/dppo` | MIT: `licenses/dppo.LICENSE` |
| Stable-Baselines3 | `vendor/dsrl/stable-baselines3` | MIT: `vendor/dsrl/stable-baselines3/LICENSE` |
| LeRobot / ACT | `vendor/lerobot` | Apache 2.0: `licenses/lerobot.LICENSE` |

The Robomimic and vector-environment code includes adaptations for the observation, reset, and action-chunk execution conventions used by ExRL. The LeRobot subset includes low-dimensional ACT and its checkpoint processors.
