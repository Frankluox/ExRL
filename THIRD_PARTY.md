# Third-party software

The bundled dependencies retain their upstream copyright and license notices.

| Component | Location | License |
| --- | --- | --- |
| Robomimic | `vendor/robomimic` | MIT: `licenses/robomimic.LICENSE` |
| DPPO / DSRL utilities | `vendor/dsrl/dppo` | MIT: `licenses/dppo.LICENSE` |
| Stable-Baselines3 | `vendor/dsrl/stable-baselines3` | MIT: `vendor/dsrl/stable-baselines3/LICENSE` |
| LeRobot / ACT | `vendor/lerobot` | Apache 2.0: `licenses/lerobot.LICENSE` |

The Robomimic and vector-environment code includes adaptations for the observation, reset, and action-chunk execution conventions used by ExRL. The LeRobot subset includes low-dimensional ACT and its checkpoint processors.

The state-based learners build on the DSRL/DPPO, Robomimic, and SB3 components above. The PA-RL, ResFiT, OGPO, and DICE-RL directories contain the adaptations described in the paper. The image-based ExRL learner adapts components from [DSRL for π₀](https://github.com/nakamotoo/dsrl_pi0), [jaxrl2](https://github.com/ikostrikov/jaxrl2), and [PTR](https://github.com/Asap7772/PTR). The jaxrl2 MIT license is retained in `licenses/jaxrl2.LICENSE`. Its discrete Q/SAC implementation is shared by LIBERO and real-robot training. The ResNet encoder retains links to its Flax and Big Transfer sources.

OpenPI, LIBERO, Cosmos Policy, ROS, and the Piper driver are installed separately and remain subject to their upstream licenses. `exrl/libero/serve_cosmos.py` retains NVIDIA's Apache-2.0 copyright and license notice.

The base-policy integrations invoke OpenPI and Cosmos training code from separately installed upstream checkouts. The LIBERO conversion follows OpenPI’s Apache-2.0 example (`licenses/openpi.LICENSE`), and `exrl/pretraining/cosmos_data.py` retains NVIDIA’s Apache-2.0 notice. Piper data transforms and the generic robot interfaces are supplied in this repository.
