"""Advanced runner options. Standard experiments use configs/robomimic/."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", type=str, required=True, help="DP checkpoint (.pt/.pth)"
    )
    parser.add_argument(
        "--vendor_root", type=str, required=True, help="Bundled environment utilities"
    )
    parser.add_argument(
        "--env_meta_json", type=str, required=True, help="robomimic env meta json path"
    )
    parser.add_argument(
        "--dsrl_normalization_npz",
        type=str,
        required=True,
        help="DSRL env normalization npz",
    )
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument(
        "--lowdim_keys",
        type=str,
        default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos,object",
    )
    parser.add_argument("--cond_steps", type=int, default=1)
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--log_dir", type=str, default="outputs")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Existing ExRL run directory to resume from latest_model.pt",
    )
    parser.add_argument("--resume_ckpt_name", type=str, default="latest_model.pt")
    parser.add_argument(
        "--resume_eval_before_training",
        action="store_true",
        help="Run one online evaluation at the restored env-step before taking any training step.",
    )
    parser.add_argument(
        "--resume_eval_min_success",
        type=float,
        default=-1.0,
        help="Abort a resume when its pre-training evaluation is below this success rate; negative disables the check.",
    )

    parser.add_argument(
        "--stats_npz_override", type=str, required=True, help="Task lowdim stats npz"
    )
    parser.add_argument(
        "--physical_obs_clip",
        type=float,
        default=10.0,
        help="Symmetric clip after physical-observation standardization; use 0 to disable.",
    )
    parser.add_argument(
        "--physical_obs_normalization",
        choices=["stats", "none"],
        default="stats",
        help="RL physical-observation transform: stats applies mean/std; none keeps DSRL-normalized state.",
    )
    parser.add_argument(
        "--action_obs_normalization",
        choices=["stats", "none"],
        default="stats",
        help="RL action-chunk/history transform: stats applies mean/std; none keeps DP-normalized actions.",
    )
    parser.add_argument("--max_chunk_len", type=int, default=32)
    parser.add_argument(
        "--min_chunk_len",
        type=int,
        default=1,
        help="Minimum legal chunk length in the action space. Use 0 to expose a zero-length resample action.",
    )
    parser.add_argument("--env_horizon_override", type=int, default=170)
    parser.add_argument("--action_history_len", type=int, default=0)
    parser.add_argument("--resample_penalty", type=float, default=-5.0)
    parser.add_argument(
        "--zero_action_fail_streak",
        type=int,
        default=0,
        help="If > 0, terminate the episode as failure after this many consecutive k=0 actions.",
    )
    parser.add_argument(
        "--zero_action_fail_reward",
        type=float,
        default=-1000.0,
        help="Terminal reward used when --zero_action_fail_streak triggers. Should be strongly negative to avoid becoming a shortcut.",
    )
    parser.add_argument("--reward_offset", type=float, default=1.0)
    parser.add_argument(
        "--terminate_on_success",
        dest="terminate_on_success",
        action="store_true",
        default=True,
        help="End episode immediately when info['is_success']=True. ",
    )
    parser.add_argument(
        "--no_terminate_on_success",
        dest="terminate_on_success",
        action="store_false",
        help="Disable success-terminal semantics and keep rolling after success.",
    )
    parser.add_argument(
        "--timeout_on_horizon",
        action="store_true",
        help="Mark horizon-end as timeout/truncation so off-policy backup bootstraps through horizon.",
    )
    parser.add_argument(
        "--reward_duration_mode",
        choices=["sum", "mean"],
        default="sum",
        help="SAC backup reward mode: sum=original chunk return, mean=chunk return divided by exec_len.",
    )
    parser.add_argument(
        "--transition_weight_mode",
        choices=["none", "duration", "sqrt_duration"],
        default="none",
        help="Optional loss weighting for replay transitions; use to counter decision-count bias without changing reward target.",
    )

    parser.add_argument("--total_policy_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--stop_after_env_steps",
        type=int,
        default=0,
        help="If > 0, stop cleanly once collected env_steps reaches this value.",
    )
    parser.add_argument("--warmup_policy_steps", type=int, default=20_000)
    parser.add_argument(
        "--warmup_action_mode", choices=["uniform", "fixed", "mixed"], default="uniform"
    )
    parser.add_argument("--warmup_fixed_k", type=int, default=12)
    parser.add_argument("--warmup_mixed_uniform_prob", type=float, default=0.3)
    parser.add_argument(
        "--epsilon_greedy_prob",
        type=float,
        default=0.0,
        help="After warmup, replace the policy-selected chunk size with a uniformly random legal chunk size with probability epsilon.",
    )
    parser.add_argument(
        "--mask_illegal_actions_with_floor",
        action="store_true",
        help="While min-k floor is active, sample and update only over legal actions >= floor. Default off preserves prior behavior.",
    )
    parser.add_argument(
        "--floor_support_mask_mode",
        choices=["none", "sample_only", "update_only", "all"],
        default="none",
        help="More precise floor-support masking. 'all' matches --mask_illegal_actions_with_floor; 'sample_only' is a lighter early-phase intervention.",
    )
    parser.add_argument(
        "--learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay SAC updates until this policy-step count (can be > warmup_policy_steps).",
    )
    parser.add_argument(
        "--probe_model_k_during_warmup",
        action="store_true",
        help="Log model-proposed k (diagnostic) during warmup without affecting executed actions.",
    )
    parser.add_argument(
        "--warmup_probe_deterministic",
        action="store_true",
        help="Use deterministic actor for warmup probe logging (default stochastic).",
    )
    parser.add_argument("--post_warmup_min_k", type=int, default=1)
    parser.add_argument("--post_warmup_min_k_until_policy_steps", type=int, default=0)
    parser.add_argument(
        "--post_warmup_min_k_schedule",
        type=str,
        default="",
        help="Optional piecewise min-k schedule, e.g. '6000:6,30000:5,50000:4,70000:3,90000:2,110000:1'.",
    )
    parser.add_argument(
        "--post_warmup_min_k_schedule_axis",
        choices=["policy", "env"],
        default="policy",
        help="Axis for explicit min-k schedule. Default keeps old behavior on policy-steps; env aligns release timing to env-step AUC.",
    )
    parser.add_argument(
        "--post_warmup_behavior_mix_schedule",
        type=str,
        default="",
        help="Optional policy/uniform behavior-mix schedule, e.g. '4000:0.7,16000:0.5,32000:0.3,60000:0.1,100000:0.0'. "
        "Only affects early data collection; empty means disabled.",
    )
    parser.add_argument(
        "--post_warmup_actor_prior_schedule",
        type=str,
        default="",
        help="Optional early actor-prior schedule, e.g. '4000:4.0,16000:3.0,30000:2.0,45000:1.0,60000:0.5,80000:0.0'. "
        "Adds a decaying KL-style prior toward larger-k support only in the early phase; empty means disabled.",
    )
    parser.add_argument(
        "--dynamic_floor_release_metric",
        choices=[
            "none",
            "critic_loss",
            "actor_loss",
            "q_argmax_k_mean",
            "success_rate_recent",
        ],
        default="none",
        help="Optional one-way trigger that permanently releases min-k floor to 1 once the chosen metric crosses threshold.",
    )
    parser.add_argument(
        "--dynamic_floor_release_threshold",
        type=float,
        default=0.0,
        help="Threshold for --dynamic_floor_release_metric; trigger fires when metric >= threshold.",
    )
    parser.add_argument(
        "--dynamic_floor_release_after_policy_steps",
        type=int,
        default=0,
        help="Do not allow dynamic floor release before this policy-step count.",
    )
    parser.add_argument(
        "--dynamic_floor_release_target_schedule",
        type=str,
        default="",
        help="Optional schedule to use after dynamic release is triggered. Empty means force floor to 1 permanently.",
    )
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument(
        "--train_freq_transitions",
        type=int,
        default=0,
        help="If > 0, trigger SAC updates every N decision-transitions collected, making update intensity less sensitive to n_env.",
    )
    parser.add_argument("--utd", type=int, default=2)
    parser.add_argument(
        "--utd_schedule",
        type=str,
        default="",
        help="Optional piecewise UTD schedule, e.g. '0:2,400000:6,600000:8'. Empty keeps fixed --utd.",
    )
    parser.add_argument(
        "--utd_schedule_axis",
        choices=["policy", "env"],
        default="env",
        help="Axis for --utd_schedule. Default env is better for fly-up phase alignment.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument(
        "--pipeline",
        choices=["online_sac", "sac_warmstart"],
        default="online_sac",
        help="Training pipeline. Default online_sac preserves existing behavior.",
    )
    parser.add_argument(
        "--warmup_transitions",
        type=int,
        default=-1,
        help="Uniform-random chunk collection steps before critic-only warmstart. Negative uses --warmup_policy_steps.",
    )
    parser.add_argument(
        "--initial_critic_updates",
        type=int,
        default=5000,
        help="Offline update count after phase0 replay collection.",
    )
    parser.add_argument(
        "--warmstart_actor",
        action="store_true",
        help="Update actor and alpha during phase1, using normal SAC actor soft-value targets. Default keeps critic-only q_greedy warmstart.",
    )
    parser.add_argument(
        "--initial_actor_updates",
        type=int,
        default=0,
        help="Optional actor-only updates after offline critic warmstart and before online SAC handoff.",
    )
    parser.add_argument(
        "--warmstart_temperature",
        action="store_true",
        help="Also update SAC alpha during optional phase2 actor-only offline updates.",
    )
    parser.add_argument(
        "--offline_eval_every_updates",
        type=int,
        default=0,
        help="Run offline eval every N phase1/phase2 updates. Set <=0 to evaluate only at phase end.",
    )
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--offline_eval_max_decisions", type=int, default=20000)
    parser.add_argument(
        "--offline_eval_policy_mode", choices=["q_greedy", "q_soft"], default="q_greedy"
    )
    parser.add_argument(
        "--offline_eval_critic_reduce",
        choices=["min", "mean", "q1", "q2"],
        default="mean",
    )
    parser.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Duration-aware SMDP n-step target. Default 1 preserves existing one-step behavior.",
    )
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--actor_learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay actor updates until this policy-step count; critic can still update earlier.",
    )
    parser.add_argument(
        "--alpha_learning_starts_policy_steps",
        type=int,
        default=-1,
        help="If > 0, delay alpha updates until this policy-step count; useful to prevent early entropy collapse.",
    )

    parser.add_argument("--n_envs", type=int, default=48)
    parser.add_argument(
        "--vecenv_start_method",
        choices=["spawn", "forkserver", "fork"],
        default="forkserver",
    )
    parser.add_argument("--num_workers_cpu_affinity", type=str, default="")

    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument(
        "--init_alpha",
        type=float,
        default=0.1,
        help="Initial entropy coefficient before alpha updates; default preserves prior behavior.",
    )
    parser.add_argument(
        "--fixed_alpha",
        type=float,
        default=None,
        help="If set, freeze alpha to this value and skip alpha optimization entirely.",
    )
    parser.add_argument(
        "--max_alpha",
        type=float,
        default=None,
        help="Optional hard cap on the learned alpha. When unset, alpha remains uncapped.",
    )
    parser.add_argument("--target_entropy_scale", type=float, default=0.98)
    parser.add_argument("--hidden_dims", type=str, default="512,512,512")
    parser.add_argument(
        "--network_activation", choices=["relu", "tanh"], default="relu"
    )
    parser.add_argument("--actor_use_layernorm", action="store_true")
    parser.add_argument("--critic_use_layernorm", action="store_true")
    parser.add_argument("--critic_backup_temp", type=float, default=1.0)
    parser.add_argument("--cql_alpha", type=float, default=0.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--critic_ensemble_size", type=int, default=2)
    parser.add_argument("--target_critic_subsample_size", type=int, default=0)
    parser.add_argument("--critic_bootstrap_mask_prob", type=float, default=1.0)
    parser.add_argument(
        "--target_q_clip_min",
        type=float,
        default=None,
        help="Optional lower clamp for TD target_q. Unset disables lower clamp.",
    )
    parser.add_argument(
        "--target_q_clip_max",
        type=float,
        default=None,
        help="Optional upper clamp for TD target_q. Unset disables upper clamp.",
    )
    parser.add_argument(
        "--actor_target_critic_subsample",
        action="store_true",
        help=(
            "When SAC online target_policy_mode=actor, subsample target critic heads "
            "before the ensemble-min target, using --target_critic_subsample_size."
        ),
    )
    parser.add_argument(
        "--actor_loss_critic_subsample",
        action="store_true",
        help=(
            "When updating the SAC actor, subsample current critic heads before "
            "the ensemble-min actor objective, using --target_critic_subsample_size."
        ),
    )
    parser.add_argument(
        "--selective_critic_forward",
        action="store_true",
        help=(
            "Evaluate only the sampled critic heads for target and actor losses. "
            "This preserves the sampled ensemble objective while avoiding unused heads."
        ),
    )
    parser.add_argument(
        "--foreach_target_update",
        action="store_true",
        help="Use fused foreach tensor operations for the target-critic soft update.",
    )
    parser.add_argument(
        "--prefetch_utd_batches",
        action="store_true",
        help=(
            "Sample and transfer an entire UTD block at once, reducing replay "
            "host-to-device launch overhead without changing sampled transitions."
        ),
    )
    parser.add_argument(
        "--async_batch_validation",
        action="store_true",
        help="Use nonblocking CUDA assertions for replay-batch validation.",
    )
    parser.add_argument(
        "--defer_utd_metric_sync",
        action="store_true",
        help="Materialize scalar metrics once after each UTD block instead of once per update.",
    )
    parser.add_argument(
        "--sparse_utd_diagnostics",
        action="store_true",
        help="Compute expensive diagnostic-only reductions on the last update of each UTD block.",
    )
    parser.add_argument(
        "--sac_cuda_graph",
        action="store_true",
        help=(
            "Experimental: capture the complete fixed-shape SAC update, including "
            "backward, Adam, alpha, and target update, in a CUDA Graph. This requires "
            "capturable Adam and is not bitwise-equivalent to the standard optimizer."
        ),
    )
    parser.add_argument(
        "--fused_optimizer",
        action="store_true",
        help=(
            "Experimental: use fused CUDA Adam for actor, critic, and alpha. It is "
            "faster but is not bitwise-equivalent to the standard optimizer."
        ),
    )

    parser.add_argument(
        "--decision_runtime", choices=["sync", "asyncqueue"], default="sync"
    )
    parser.add_argument(
        "--dsrl_variable_k_runtime",
        choices=["parent_loop", "worker_chunk"],
        default="parent_loop",
    )
    parser.add_argument("--perf_profile", action="store_true")
    parser.add_argument("--dsrl_prune_unused_observables", action="store_true")
    parser.add_argument("--dsrl_worker_cpu_ids", type=str, default="")
    parser.add_argument("--dsrl_cache_static_robot_metadata", action="store_true")
    parser.add_argument("--dp_infer_steps", type=int, default=5)
    parser.add_argument("--dp_sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--dp_amp", action="store_true")
    parser.add_argument("--dp_compile", action="store_true")
    parser.add_argument("--dp_cuda_graph", action="store_true")
    parser.add_argument("--dp_infer_parallel", action="store_true")
    parser.add_argument("--dp_infer_gpus", type=str, default="0")
    parser.add_argument(
        "--dp_grouping", choices=["round_robin", "contiguous"], default="round_robin"
    )
    parser.add_argument("--ipc_chunk_dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument(
        "--ipc_send_mode", choices=["full", "executed_only"], default="executed_only"
    )

    parser.add_argument("--infer_microbatch_max", type=int, default=64)
    parser.add_argument("--infer_microbatch_timeout_ms", type=int, default=3)
    parser.add_argument("--infer_queue_size", type=int, default=4096)
    parser.add_argument("--infer_result_queue_size", type=int, default=4096)
    parser.add_argument("--env_poll_timeout_ms", type=int, default=5)
    parser.add_argument("--ready_queue_size", type=int, default=8192)
    parser.add_argument("--max_inflight_per_env", type=int, default=1)
    parser.add_argument("--async_rollout_mode", type=str, default="strict_on_policy")

    parser.add_argument("--log_every_policy_steps", type=int, default=2048)
    parser.add_argument(
        "--online_eval_every_env_steps",
        type=int,
        default=30_000,
        help="Run greedy actor evaluation every N env steps. Set <=0 to disable.",
    )
    parser.add_argument(
        "--online_eval_episodes",
        type=int,
        default=50,
        help="Number of episodes for each online greedy actor evaluation.",
    )
    parser.add_argument(
        "--online_eval_n_envs",
        type=int,
        default=10,
        help="Parallel eval env count for online evaluation. Values <=0 use min(train n_envs, episodes).",
    )
    parser.add_argument(
        "--online_eval_max_decisions_per_episode",
        type=int,
        default=1000,
        help="Safety cap for greedy eval decisions per episode, mainly to avoid infinite k=0 loops.",
    )
    parser.add_argument(
        "--eval_snapshot_env_freq",
        type=int,
        default=0,
        help=(
            "Write an actor-only snapshot every N env steps for an external evaluator. "
            "Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--debug_cadence_metrics",
        action="store_true",
        help="Print and log collector/replay/update cadence counters for short smoke runs.",
    )
    parser.add_argument("--metrics_flush_every_logs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=200_000)
    parser.add_argument("--latest_model_save_freq", type=int, default=200_000)

    parser.add_argument("--force_thread_limits", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    return parser
