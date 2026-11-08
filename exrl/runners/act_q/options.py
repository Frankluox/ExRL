"""Advanced runner options. Standard experiments use configs/robomimic/."""

import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="ACT checkpoint dir or ACT training output dir",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="ACT task preset: lift/can/square/transport",
    )
    parser.add_argument(
        "--vendor_root", type=str, required=True, help="Bundled environment utilities"
    )
    parser.add_argument(
        "--env_meta_json", type=str, default=None, help="robomimic env meta json path"
    )
    parser.add_argument(
        "--dsrl_normalization_npz",
        type=str,
        default=None,
        help="DSRL env normalization npz",
    )
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--lowdim_keys", type=str, default="")
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
        "--stats_npz_override", type=str, default=None, help="Task lowdim stats npz"
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
        choices=["online_sac", "q_learning"],
        default="online_sac",
        help="SAC or Q-learning with random warmup followed by critic initialization.",
    )
    parser.add_argument(
        "--warmup_transitions",
        type=int,
        default=-1,
        help="Phase-0 random collection policy steps. If <=0, reuse --warmup_policy_steps.",
    )
    parser.add_argument(
        "--initial_critic_updates",
        type=int,
        default=100_000,
        help="Number of critic-only updates after phase-0 collection.",
    )
    parser.add_argument(
        "--online_transitions",
        type=int,
        default=0,
        help=(
            "If >0, continue after phase-1 with critic-policy rollout plus "
            "critic-only updates for this many policy decisions."
        ),
    )
    parser.add_argument(
        "--collection_policy",
        choices=["q_greedy", "q_soft", "q_ucb", "q_thompson"],
        default="q_greedy",
        help="How the critic selects behavior actions during phase-2 rollout.",
    )
    parser.add_argument(
        "--collection_critic_reduce",
        choices=["min", "mean"],
        default="mean",
        help="How phase-2 behavior reduces the critic ensemble before selecting k.",
    )
    parser.add_argument(
        "--collection_epsilon",
        type=float,
        default=0.0,
        help="During phase-2, replace critic-selected k with uniform random legal k with this probability.",
    )
    parser.add_argument(
        "--collection_ucb_beta",
        type=float,
        default=0.0,
        help="UCB exploration beta for --collection_policy q_ucb.",
    )
    parser.add_argument(
        "--collection_ucb_schedule",
        type=str,
        default="",
        help="Env-step schedule for phase-2 q_ucb beta, e.g. '0:1.0,1000000:0.3,3000000:0.0'.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=30_000,
        help=(
            "Offline pipeline env-step eval interval on the training env-step x-axis. "
            "0 disables this eval track. Phase-1 critic-only updates do not advance this schedule."
        ),
    )
    parser.add_argument(
        "--offline_resume_save_interval_steps",
        type=int,
        default=0,
        help=(
            "Extra offline latest checkpoint interval for robust resume. If 0, phase0/2 "
            "reuse --eval_interval when enabled, and phase1 reuses "
            "--offline_eval_every_updates when enabled."
        ),
    )
    parser.add_argument(
        "--offline_async_eval_snapshot_interval_steps",
        type=int,
        default=0,
        help=(
            "Write an atomic critic + current base-policy snapshot every this many env steps for an "
            "external evaluator. 0 disables snapshots and preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--eval_at_start",
        dest="eval_at_start",
        action="store_true",
        default=True,
        help="Run env-step eval once at training env_step=0 before phase-0 random collection.",
    )
    parser.add_argument(
        "--no_eval_at_start",
        dest="eval_at_start",
        action="store_false",
        help="Disable the initial env_step=0 env-step eval.",
    )
    parser.add_argument(
        "--offline_eval_every_updates",
        type=int,
        default=5_000,
        help="Run critic-as-actor eval every this many phase-1 updates. 0 disables eval.",
    )
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument(
        "--offline_eval_max_decisions",
        type=int,
        default=20000,
        help="Safety cap on critic-policy eval decision steps.",
    )
    parser.add_argument(
        "--offline_eval_policy_mode",
        choices=["q_greedy", "q_soft"],
        default="q_greedy",
        help="How critic chooses actions during phase-1 eval.",
    )
    parser.add_argument(
        "--offline_eval_critic_reduce",
        choices=["min", "mean"],
        default="mean",
        help="How eval reduces the critic ensemble before selecting k. PA-RL optimized eval uses mean-like ranking.",
    )
    parser.add_argument(
        "--eval_n_envs",
        type=int,
        default=-1,
        help="Number of envs for the separate offline eval env. <=0 uses min(n_envs, eval_episodes).",
    )
    parser.add_argument(
        "--offline_eval_seed_offset",
        type=int,
        default=10000,
        help="Seed offset for the separate offline eval env.",
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
    parser.add_argument(
        "--target_policy_mode",
        choices=["actor", "q_soft", "q_greedy"],
        default="actor",
        help="SMDP target policy. SAC uses the actor; Q-learning uses the critic.",
    )
    parser.add_argument("--critic_backup_temp", type=float, default=1.0)
    parser.add_argument("--cql_alpha", type=float, default=0.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--use_mc_lower_bound", action="store_true")
    parser.add_argument(
        "--mc_lower_bound_scope",
        choices=["all", "non_data", "data", "topk_non_data"],
        default="topk_non_data",
        help="Where MC lower bound is applied inside CQL. topk_non_data is closest to PA-RL CalQL.",
    )
    parser.add_argument(
        "--mc_lower_bound_topk",
        type=int,
        default=10,
        help="Top-k action count for --mc_lower_bound_scope topk_non_data.",
    )
    parser.add_argument("--critic_ensemble_size", type=int, default=2)
    parser.add_argument("--target_critic_subsample_size", type=int, default=0)
    parser.add_argument("--critic_bootstrap_mask_prob", type=float, default=1.0)
    parser.add_argument("--distributional_critic", action="store_true")
    parser.add_argument("--distributional_q_min", type=float, default=-300.0)
    parser.add_argument("--distributional_q_max", type=float, default=0.0)
    parser.add_argument("--distributional_num_bins", type=int, default=128)
    parser.add_argument(
        "--td_loss_type",
        choices=["mse", "huber"],
        default="mse",
        help="Critic TD regression loss. Huber is a simple guard against large TD explosions.",
    )
    parser.add_argument(
        "--td_huber_delta",
        type=float,
        default=10.0,
        help="Beta/delta for --td_loss_type huber.",
    )
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
        "--critic_q_upper_bound",
        type=float,
        default=None,
        help="Optional upper bound used by a soft Q-cap penalty on all critic outputs.",
    )
    parser.add_argument(
        "--critic_q_upper_penalty",
        type=float,
        default=0.0,
        help="Weight for relu(Q - critic_q_upper_bound)^2. 0 disables the Q-cap penalty.",
    )
    parser.add_argument(
        "--td_target_next_obs_samples",
        type=int,
        default=1,
        help=(
            "Number of independently sampled base-policy chunks used for TD target next-state averaging. "
            "1 preserves the old single-sample behavior."
        ),
    )
    parser.add_argument(
        "--critic_obs_unnormalize_dp_chunk",
        action="store_true",
        help=(
            "Compatibility alias for --critic_obs_action_mode dsrl_unnorm_stats. "
            "It affects only the critic/RL observation, not ACT execution."
        ),
    )
    parser.add_argument(
        "--critic_obs_action_mode",
        choices=["stats", "dsrl_unnorm_stats", "dp_raw"],
        default="stats",
        help=(
            "Base action chunk representation used by the critic. stats preserves normalized ACT chunks; "
            "dsrl_unnorm_stats maps DSRL-normalized actions to physical coordinates before z-score; "
            "dp_raw feeds the ACT base-policy output chunk directly."
        ),
    )
    parser.add_argument(
        "--critic_obs_state_mode",
        choices=["stats_clip", "stats_no_clip", "dsrl_raw"],
        default="stats_clip",
        help=(
            "State representation used by the critic. stats_clip preserves normalized ACT chunks; "
            "stats_no_clip removes only the final [-10,10] clamp; dsrl_raw feeds the "
            "DSRL-normalized state directly."
        ),
    )

    parser.add_argument(
        "--bc_finetune_mode",
        choices=["off", "online_success"],
        default="off",
        help="Optionally finetune the ACT base policy on successful online episodes.",
    )
    parser.add_argument("--bc_updates_per_rl_update", type=int, default=1)
    parser.add_argument("--bc_updates_final", type=float, default=1.0)
    parser.add_argument(
        "--bc_updates_decay_env_steps",
        type=int,
        default=0,
        help="Linear BC update decay duration after BC first becomes eligible; 0 disables decay.",
    )
    parser.add_argument("--bc_update_every_rl_updates", type=int, default=2)
    parser.add_argument("--bc_update_every_rl_updates_schedule", type=str, default="")
    parser.add_argument("--bc_schedule_axis", choices=["policy", "env"], default="env")
    parser.add_argument("--bc_start_after_steps", type=int, default=0)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-5)
    parser.add_argument("--bc_lr_schedule", type=str, default="")
    parser.add_argument(
        "--bc_act_trainable_scope",
        choices=["all", "decoder_head", "action_head"],
        default="all",
        help=(
            "ACT parameters updated by online BC. all preserves existing behavior; "
            "decoder_head freezes the encoder and input projections; action_head updates only "
            "the final action regression layer."
        ),
    )
    parser.add_argument(
        "--bc_grad_clip_norm",
        type=float,
        default=0.0,
        help="Clip ACT online-BC gradient norm before each optimizer step; 0 preserves legacy no-clipping behavior.",
    )
    parser.add_argument("--bc_success_capacity", type=int, default=25000)
    parser.add_argument(
        "--bc_online_sampling", choices=["window", "episode"], default="episode"
    )
    parser.add_argument("--bc_offline_demo_npz", type=str, default="")
    parser.add_argument("--bc_offline_fraction", type=float, default=0.0)
    parser.add_argument(
        "--bc_offline_sampling", choices=["window", "episode"], default="episode"
    )
    parser.add_argument(
        "--bc_offline_start_mode",
        choices=["match_online", "immediate"],
        default="match_online",
        help="match_online preserves the original full-online-batch readiness gate for all BC mixtures.",
    )
    parser.add_argument(
        "--max_train_env_steps",
        type=int,
        default=0,
        help="Finish phase2 normally at this total primitive interaction budget (including warmup); 0 disables.",
    )
    parser.add_argument(
        "--diagnostic_snapshot_env_steps",
        type=str,
        default="",
        help="Comma-separated primitive-step thresholds for retained ACT+critic snapshots, without replay.",
    )
    parser.add_argument(
        "--bc_tail_padding",
        choices=["act_native", "unpadded"],
        default="act_native",
        help=(
            "ACT-native keeps every successful-episode state, repeat-pads the tail action chunk, "
            "and masks padded targets exactly like ACT pretraining. Unpadded is retained for "
            "compatibility with early ACT online-BC checkpoints."
        ),
    )
    parser.add_argument(
        "--bc_teacher_constraint_weight",
        type=float,
        default=1.0,
        help="MSE weight that anchors deterministic ACT outputs to the initial or EMA teacher.",
    )
    parser.add_argument(
        "--bc_teacher_ema_decay",
        type=float,
        default=1.0,
        help="EMA decay for the teacher; 1.0 keeps the initial ACT checkpoint as a fixed teacher.",
    )
    parser.add_argument(
        "--allow_bc_teacher_weight_change_on_resume",
        action="store_true",
        help="Explicitly allow changing the ACT teacher constraint weight when resuming.",
    )
    parser.add_argument(
        "--allow_bc_teacher_ema_decay_change_on_resume",
        action="store_true",
        help="Explicitly allow changing the ACT teacher EMA decay when resuming.",
    )

    parser.add_argument(
        "--decision_runtime", choices=["sync", "asyncqueue"], default="sync"
    )
    parser.add_argument("--dp_infer_steps", type=int, default=5)
    parser.add_argument("--dp_sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--dp_amp", action="store_true")
    parser.add_argument("--dp_compile", action="store_true")
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
        "--debug_cadence_metrics",
        action="store_true",
        help="Print and log collector/replay/update cadence counters for short smoke runs.",
    )
    parser.add_argument("--metrics_flush_every_logs", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=200_000)
    parser.add_argument("--latest_model_save_freq", type=int, default=200_000)
    best_model_group = parser.add_mutually_exclusive_group()
    best_model_group.add_argument(
        "--save_best_model",
        dest="save_best_model",
        action="store_true",
        help="Save an overwriting best_model.pt whenever rollout success improves (default).",
    )
    best_model_group.add_argument(
        "--no_save_best_model",
        dest="save_best_model",
        action="store_false",
        help="Track best success in checkpoint state without writing best_model.pt.",
    )
    parser.set_defaults(save_best_model=True)
    parser.add_argument("--disable_checkpoints", action="store_true")

    parser.add_argument("--force_thread_limits", action="store_true")
    parser.add_argument("--save_videos", action="store_true")
    return parser
