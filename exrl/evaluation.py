"""Evaluation of full or compact ExRL checkpoints, including an adapted base policy."""

from __future__ import annotations
import inspect
import numpy as np
import torch


def evaluate(trainer, args, recipe, checkpoint, episodes):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    from importlib import import_module
    from exrl.config import runner_name

    name = runner_name(recipe)
    make_env = import_module(f"exrl.runners.{name}.evaluation_env").make_env
    core = import_module(f"exrl.algorithms.{name}")
    trainer._set_runtime_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    keys = args.lowdim_keys.split(",")
    dims = trainer.infer_lowdim_dims_from_stats(args.stats_npz_override, keys)
    if recipe["base_policy"] == "act":
        policy, _ = trainer.load_act_policy_wrapper(
            ckpt_path=args.agent, device=device, task=recipe["task"]
        )
    else:
        policy, _ = trainer.load_dsrl_policy_wrapper(
            ckpt_path=args.agent,
            device=device,
            stats_npz=args.stats_npz_override,
            lowdim_keys=keys,
            lowdim_dims=dims,
            action_dim=trainer.infer_action_dim_from_stats(args.stats_npz_override),
            cond_steps=args.cond_steps,
            weight_source=args.weight_source,
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if recipe["base_policy"] == "act":
        base = state.get("act_policy_state")
        if base is not None:
            policy.policy.policy_model.load_state_dict(base, strict=True)
    else:
        base = state.get("bc_live_policy_state", state.get("base_policy_state"))
        if base is not None:
            load_module_state_dict_compat = import_module(
                "exrl.runners.dp_q.online_bc"
            ).load_module_state_dict_compat
            load_module_state_dict_compat(policy.policy, base, strict=True)
    if (
        str(getattr(args, "bc_finetune_mode", "off")) not in ("none", "off", "")
        and base is None
    ):
        raise ValueError(
            "The BC recipe requires a checkpoint containing the adapted base policy."
        )
    env_meta = trainer.load_env_meta_json(args.env_meta_json)
    env_meta.setdefault("env_kwargs", {})["horizon"] = args.env_horizon_override
    args.dp_compile = False
    args.dp_cuda_graph = False
    args.dp_infer_gpus = ""
    env = make_env(
        args=args,
        policy=policy,
        device=device,
        lowdim_keys=keys,
        lowdim_dims=dims,
        env_meta=env_meta,
        n_envs=min(4, episodes),
        seed=args.seed + 10000,
    )
    try:
        obs = np.asarray(env.reset(), dtype=np.float32)
        action_dim = args.max_chunk_len - args.min_chunk_len + 1
        raw_dim = env.action_space.n
        if raw_dim == action_dim:
            shift = 0
        elif raw_dim == args.max_chunk_len + 1:
            shift = args.min_chunk_len
        else:
            raise ValueError("Execution-length action space does not match the recipe")
        hidden = tuple(int(v) for v in args.hidden_dims.split(","))
        if "actor_state" in state and "sac_state" not in state:
            actor = core.DiscreteActor(
                obs.shape[1],
                action_dim,
                state.get("hidden_dims", hidden),
                activation=state.get("activation", args.network_activation),
                use_layernorm=state.get(
                    "actor_use_layernorm", args.actor_use_layernorm
                ),
            ).to(device)
            actor.load_state_dict(state["actor_state"], strict=True)
            actor.eval()

            def act(x):
                with torch.no_grad():
                    return (
                        actor(torch.as_tensor(x, device=device))
                        .argmax(-1)
                        .cpu()
                        .numpy()
                    )

        else:
            weights = state.get("sac_state", state)
            opts = {
                **vars(args),
                **{
                    k: v
                    for k, v in weights.items()
                    if not isinstance(v, (dict, torch.Tensor))
                },
            }
            opts.update(
                obs_dim=obs.shape[1],
                action_dim=action_dim,
                hidden_dims=hidden,
                device=device,
                activation=args.network_activation,
                actor_enabled=recipe["algorithm"] == "sac",
            )
            if not opts["actor_enabled"]:
                opts["target_policy_mode"] = "q_greedy"
            opts = {
                k: v
                for k, v in opts.items()
                if k in inspect.signature(core.ExecutionLengthAgent).parameters
            }
            model = core.ExecutionLengthAgent(**opts)
            critic = weights.get("critic", weights.get("critic_state"))
            if critic is None:
                raise ValueError("Missing critic weights in checkpoint")
            model.critic.load_state_dict(critic, strict=True)
            model.critic.eval()
            if opts["actor_enabled"]:
                model.actor.load_state_dict(weights["actor"], strict=True)
                model.actor.eval()

                def act(x):
                    return model.select_action(x, deterministic=True, min_action_id=0)

            else:

                def act(x):
                    return model.select_critic_action(
                        x,
                        deterministic=True,
                        min_action_id=0,
                        mode="q_greedy",
                        reduce=getattr(args, "offline_eval_critic_reduce", "mean"),
                    )

        successes = []
        lengths = []
        env_steps = 0
        decisions = 0
        limit = episodes * max(args.env_horizon_override, 1) * 4
        while len(successes) < episodes and decisions < limit:
            obs, _, dones, infos = env.step(act(obs) + shift)
            obs = np.asarray(obs, dtype=np.float32)
            decisions += env.num_envs
            for done, info in zip(dones, infos):
                k = int(info.get("exec_len", 0))
                lengths.append(k)
                env_steps += k
                if done and len(successes) < episodes:
                    successes.append(bool(trainer._extract_success(info)))
        if len(successes) < episodes:
            raise RuntimeError(
                f"Only {len(successes)}/{episodes} episodes finished within the decision budget"
            )
        return dict(
            episodes=episodes,
            success_rate=float(np.mean(successes)),
            mean_execution_length=float(np.mean(lengths)),
            environment_steps=env_steps,
            seed=args.seed,
        )
    finally:
        env.close()
