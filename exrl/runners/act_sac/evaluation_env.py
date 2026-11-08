from exrl.runners.act_sac.train import (
    ExecutionLengthEnv,
    VecMonitor,
    _parse_cpu_affinity,
)


def make_env(*, args, policy, device, lowdim_keys, lowdim_dims, env_meta, n_envs, seed):
    return VecMonitor(
        ExecutionLengthEnv(
            venv=None,
            policy=policy,
            max_chunk_len=args.max_chunk_len,
            device=device,
            reward_offset=args.reward_offset,
            resample_penalty=args.resample_penalty,
            action_history_len=args.action_history_len,
            physical_obs_normalization=args.physical_obs_normalization,
            action_obs_normalization=args.action_obs_normalization,
            physical_obs_clip=args.physical_obs_clip,
            dp_infer_steps=args.dp_infer_steps,
            dp_sampler=args.dp_sampler,
            dp_amp=args.dp_amp,
            dp_compile=args.dp_compile,
            ipc_chunk_dtype=args.ipc_chunk_dtype,
            dp_infer_gpus=args.dp_infer_gpus,
            dp_grouping=args.dp_grouping,
            dp_infer_parallel=args.dp_infer_parallel,
            ipc_send_mode=args.ipc_send_mode,
            perf_profile=False,
            vendor_root=args.vendor_root,
            env_name=args.env_name,
            env_meta_json=args.env_meta_json,
            normalization_path=args.dsrl_normalization_npz,
            stats_npz_path=args.stats_npz_override,
            lowdim_keys=lowdim_keys,
            lowdim_dims=lowdim_dims,
            n_envs=n_envs,
            max_episode_steps=int(
                env_meta.get("env_kwargs", {}).get("horizon", args.env_horizon_override)
            ),
            gamma=args.gamma,
            seed=seed,
            min_chunk_len=int(args.min_chunk_len),
            terminate_on_success=bool(args.terminate_on_success),
            timeout_on_horizon=bool(args.timeout_on_horizon),
        )
    )
