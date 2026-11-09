import hashlib
import io
import json
import os
import random
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import recursive_getattr


CHECKPOINT_VERSION = 1
MODEL_STATE_ENTRY = "model_state.pt"
MANIFEST_ENTRY = "manifest.json"
DEFAULT_INTERVAL_ENV_STEPS = 100_000
REPLAY_CHUNK_BYTES = 32 * 1024 * 1024


class ResumeCheckpointError(RuntimeError):
    pass


def _torch_load(file_obj, map_location):
    try:
        return torch.load(file_obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(file_obj, map_location=map_location)


def _fsync_file(path):
    with open(path, "rb") as file_obj:
        os.fsync(file_obj.fileno())


def _fsync_directory(path):
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for block in iter(lambda: file_obj.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _config_value(cfg, path, default=None):
    value = OmegaConf.select(cfg, path, default=default)
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def config_signature(cfg):
    diffusion_horizon = int(
        _config_value(cfg, "diffusion_horizon_steps", _config_value(cfg, "act_steps"))
    )
    execute_steps = int(_config_value(cfg, "execute_steps", _config_value(cfg, "act_steps")))
    critical_paths = (
        "algorithm",
        "env_name",
        "obs_dim",
        "action_dim",
        "act_steps",
        "env.n_envs",
        "env.max_episode_steps",
        "base_policy_path",
        "normalization_path",
        "model._target_",
        "model.network.horizon_steps",
        "model.network.time_dim",
        "model.network.mlp_dims",
        "model.network.residual_style",
        "model.denoising_steps",
        "model.ddim_steps",
        "train.action_magnitude",
        "train.actor_lr",
        "train.batch_size",
        "train.discount",
        "train.ent_coef",
        "train.layer_size",
        "train.n_critics",
        "train.noise_critic_grad_steps",
        "train.num_layers",
        "train.target_ent",
        "train.tau",
        "train.train_freq",
        "train.use_layer_norm",
        "train.utd",
    )
    signature = {path: _config_value(cfg, path) for path in critical_paths}
    signature["diffusion_horizon_steps"] = diffusion_horizon
    signature["execute_steps"] = execute_steps
    return signature


def _capture_rng_state():
    cuda_states = []
    if torch.cuda.is_available():
        cuda_states = [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        cuda_states = [cuda_state.cpu() for cuda_state in state["torch_cuda"]]
        if len(cuda_states) != torch.cuda.device_count():
            raise ResumeCheckpointError(
                "CUDA device count changed across resume: "
                f"checkpoint={len(cuda_states)} current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _capture_deque(value):
    if value is None:
        return None
    return {"items": list(value), "maxlen": value.maxlen}


def _restore_deque(value):
    if value is None:
        return None
    return deque(value["items"], maxlen=value["maxlen"])


def capture_logging_callback_state(callback):
    return {
        "n_calls": int(callback.n_calls),
        "num_timesteps": int(callback.num_timesteps),
        "episode_rewards": list(callback.episode_rewards),
        "episode_lengths": list(callback.episode_lengths),
        "log_count": int(callback.log_count),
        "total_reward": callback.total_reward,
        "total_timesteps": int(callback.total_timesteps),
        "episode_success": np.asarray(callback.episode_success).copy(),
        "episode_completed": np.asarray(callback.episode_completed).copy(),
        "completed_success_recent": _capture_deque(callback.completed_success_recent),
        "completed_success_recent20": _capture_deque(callback.completed_success_recent20),
        "completed_success_since_log": list(callback.completed_success_since_log),
        "next_online_eval_env_step": int(callback.next_online_eval_env_step),
    }


def restore_logging_callback_state(callback, state):
    callback.n_calls = int(state["n_calls"])
    callback.num_timesteps = int(state["num_timesteps"])
    callback.episode_rewards = list(state["episode_rewards"])
    callback.episode_lengths = list(state["episode_lengths"])
    callback.log_count = int(state["log_count"])
    callback.total_reward = state["total_reward"]
    callback.total_timesteps = int(state["total_timesteps"])
    # The subprocess environments restart from fresh episodes, so per-env partial
    # episode flags cannot be carried across the boundary.
    callback.episode_success = np.zeros_like(np.asarray(state["episode_success"]))
    callback.episode_completed = np.zeros_like(np.asarray(state["episode_completed"]))
    callback.completed_success_recent = _restore_deque(state["completed_success_recent"])
    callback.completed_success_recent20 = _restore_deque(state["completed_success_recent20"])
    callback.completed_success_since_log = list(state["completed_success_since_log"])
    callback.next_online_eval_env_step = int(state["next_online_eval_env_step"])
    callback.online_eval_process = None


def _capture_model_state(
    model, logging_callback, numbered_checkpoint_callback, latest_state, training_state
):
    runtime_attributes = (
        "num_timesteps",
        "env_steps_exact",
        "_n_updates",
        "_episode_num",
        "_current_progress_remaining",
        "_total_timesteps",
        "_num_timesteps_at_start",
        "_num_env_steps_at_start",
    )
    runtime = {name: getattr(model, name) for name in runtime_attributes if hasattr(model, name)}
    runtime["ep_info_buffer"] = _capture_deque(getattr(model, "ep_info_buffer", None))
    runtime["ep_success_buffer"] = _capture_deque(getattr(model, "ep_success_buffer", None))

    _, torch_variable_names = model._get_torch_save_params()
    torch_variables = {
        name: recursive_getattr(model, name).detach().clone()
        for name in torch_variable_names
        if recursive_getattr(model, name) is not None
    }
    extra_optimizers = {}
    if (
        hasattr(model, "critic_noise")
        and getattr(model.critic_noise, "optimizer", None) is not None
    ):
        extra_optimizers["critic_noise.optimizer"] = model.critic_noise.optimizer.state_dict()

    return {
        "version": CHECKPOINT_VERSION,
        "parameters": model.get_parameters(),
        "torch_variables": torch_variables,
        "extra_optimizers": extra_optimizers,
        "runtime": runtime,
        "rng": _capture_rng_state(),
        "logging_callback": capture_logging_callback_state(logging_callback),
        "numbered_checkpoint_callback": {
            "n_calls": int(numbered_checkpoint_callback.n_calls),
            "num_timesteps": int(numbered_checkpoint_callback.num_timesteps),
        },
        "latest_checkpoint_callback": latest_state,
        "training": training_state,
    }


def _restore_model_state(model, state):
    model.set_parameters(state["parameters"], exact_match=True, device=model.device)
    for name, optimizer_state in state.get("extra_optimizers", {}).items():
        recursive_getattr(model, name).load_state_dict(optimizer_state)
    for name, value in state.get("torch_variables", {}).items():
        current = recursive_getattr(model, name)
        current.data.copy_(value.to(current.device))
    for name, value in state["runtime"].items():
        if name not in ("ep_info_buffer", "ep_success_buffer"):
            setattr(model, name, value)
    model.ep_info_buffer = _restore_deque(state["runtime"].get("ep_info_buffer"))
    model.ep_success_buffer = _restore_deque(state["runtime"].get("ep_success_buffer"))
    # A subprocess environment cannot be snapshotted. Force a clean episode boundary.
    model._last_obs = None
    model._last_original_obs = None
    model._last_episode_starts = None


def _array_semantic_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _save_replay_chunk(chunk, sidecar_dir):
    digest = _array_semantic_hash(chunk)
    final_path = sidecar_dir / f"{digest}.npy"
    if final_path.exists():
        return digest, final_path
    temp_path = sidecar_dir / f".{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "wb") as file_obj:
            np.save(file_obj, chunk, allow_pickle=False)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, final_path)
        _fsync_directory(sidecar_dir)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return digest, final_path


def _capture_replay_manifest(replay_buffer, sidecar_dir):
    required = ("observations", "actions", "rewards", "dones", "timeouts")
    for name in required:
        if not hasattr(replay_buffer, name):
            raise ResumeCheckpointError(f"Unsupported replay buffer: missing {name}")

    size = int(replay_buffer.buffer_size if replay_buffer.full else replay_buffer.pos)
    array_names = ["observations", "actions", "rewards", "dones", "timeouts"]
    if not replay_buffer.optimize_memory_usage:
        array_names.append("next_observations")
    if getattr(replay_buffer, "use_noise_actions", False):
        array_names.append("noise_actions")

    arrays = {}
    for name in array_names:
        array = getattr(replay_buffer, name)
        valid_rows = size
        if (
            name == "observations"
            and replay_buffer.optimize_memory_usage
            and not replay_buffer.full
        ):
            valid_rows = min(size + 1, replay_buffer.buffer_size)
        row_bytes = max(1, int(array[0].nbytes) if len(array) else int(array.dtype.itemsize))
        rows_per_chunk = max(1, REPLAY_CHUNK_BYTES // row_bytes)
        chunks = []
        for start in range(0, valid_rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, valid_rows)
            chunk = array[start:stop]
            digest, chunk_path = _save_replay_chunk(chunk, sidecar_dir)
            chunks.append(
                {
                    "start": start,
                    "stop": stop,
                    "sha256": digest,
                    "file": chunk_path.name,
                    "shape": list(chunk.shape),
                    "dtype": str(chunk.dtype),
                }
            )
        arrays[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "valid_rows": valid_rows,
            "chunks": chunks,
        }

    return {
        "buffer_size": int(replay_buffer.buffer_size),
        "n_envs": int(replay_buffer.n_envs),
        "pos": int(replay_buffer.pos),
        "full": bool(replay_buffer.full),
        "size": size,
        "optimize_memory_usage": bool(replay_buffer.optimize_memory_usage),
        "handle_timeout_termination": bool(replay_buffer.handle_timeout_termination),
        "offline_mix_ratio": int(replay_buffer.offline_mix_ratio),
        "offline_steps": int(replay_buffer.offline_steps),
        "use_noise_actions": bool(getattr(replay_buffer, "use_noise_actions", False)),
        "arrays": arrays,
    }


def _restore_replay_buffer(replay_buffer, replay_manifest, sidecar_dir):
    if int(replay_buffer.n_envs) != int(replay_manifest["n_envs"]):
        raise ResumeCheckpointError(
            f"Replay n_envs mismatch: checkpoint={replay_manifest['n_envs']} current={replay_buffer.n_envs}"
        )
    if bool(replay_buffer.optimize_memory_usage) != bool(replay_manifest["optimize_memory_usage"]):
        raise ResumeCheckpointError("Replay optimize_memory_usage changed across resume")
    if replay_manifest["full"] and int(replay_buffer.buffer_size) != int(
        replay_manifest["buffer_size"]
    ):
        raise ResumeCheckpointError(
            "A full replay buffer requires the same buffer capacity on resume"
        )
    if int(replay_buffer.buffer_size) < int(replay_manifest["size"]):
        raise ResumeCheckpointError("Current replay buffer is smaller than saved replay data")
    if replay_buffer.pos != 0 or replay_buffer.full:
        raise ResumeCheckpointError("Replay restore requires a newly initialized empty buffer")

    for name, array_manifest in replay_manifest["arrays"].items():
        if not hasattr(replay_buffer, name):
            raise ResumeCheckpointError(f"Current replay buffer is missing array {name}")
        target = getattr(replay_buffer, name)
        if str(target.dtype) != array_manifest["dtype"]:
            raise ResumeCheckpointError(
                f"Replay dtype mismatch for {name}: checkpoint={array_manifest['dtype']} current={target.dtype}"
            )
        if tuple(target.shape[1:]) != tuple(array_manifest["shape"][1:]):
            raise ResumeCheckpointError(
                f"Replay shape mismatch for {name}: checkpoint={array_manifest['shape']} current={target.shape}"
            )
        for chunk_manifest in array_manifest["chunks"]:
            chunk_path = sidecar_dir / chunk_manifest["file"]
            if not chunk_path.is_file():
                raise ResumeCheckpointError(f"Missing replay chunk: {chunk_path}")
            chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
            if (
                list(chunk.shape) != chunk_manifest["shape"]
                or str(chunk.dtype) != chunk_manifest["dtype"]
            ):
                raise ResumeCheckpointError(f"Invalid replay chunk metadata: {chunk_path}")
            if _array_semantic_hash(chunk) != chunk_manifest["sha256"]:
                raise ResumeCheckpointError(f"Replay chunk checksum mismatch: {chunk_path}")
            target[chunk_manifest["start"] : chunk_manifest["stop"]] = chunk

    replay_buffer.pos = int(replay_manifest["pos"])
    replay_buffer.full = bool(replay_manifest["full"])
    replay_buffer.handle_timeout_termination = bool(replay_manifest["handle_timeout_termination"])
    replay_buffer.offline_mix_ratio = int(replay_manifest["offline_mix_ratio"])
    replay_buffer.offline_steps = int(replay_manifest["offline_steps"])
    replay_buffer.use_noise_actions = bool(replay_manifest["use_noise_actions"])


def _validate_checkpoint_file(path, expected_model_size, expected_model_sha256):
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if not {MODEL_STATE_ENTRY, MANIFEST_ENTRY}.issubset(names):
            raise ResumeCheckpointError(f"Incomplete checkpoint archive: {path}")
        model_info = archive.getinfo(MODEL_STATE_ENTRY)
        if model_info.file_size != expected_model_size:
            raise ResumeCheckpointError("Model state size changed while building checkpoint")
        manifest = json.loads(archive.read(MANIFEST_ENTRY))
        if manifest["model_state"]["sha256"] != expected_model_sha256:
            raise ResumeCheckpointError("Model state checksum missing from manifest")
        digest = hashlib.sha256()
        with archive.open(MODEL_STATE_ENTRY, "r") as model_file:
            for block in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_model_sha256:
            raise ResumeCheckpointError("Model state checksum changed while building checkpoint")
        if manifest["version"] != CHECKPOINT_VERSION:
            raise ResumeCheckpointError("Checkpoint version validation failed")


def save_training_checkpoint(
    path,
    model,
    logging_callback,
    numbered_checkpoint_callback,
    latest_state,
    training_state,
    cfg,
):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir = path.parent / f"{path.name}.replay"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    model_temp = path.parent / f".{path.name}.{token}.model.tmp"
    checkpoint_temp = path.parent / f".{path.name}.{token}.tmp"
    try:
        model_state = _capture_model_state(
            model,
            logging_callback,
            numbered_checkpoint_callback,
            latest_state,
            training_state,
        )
        torch.save(model_state, model_temp)
        _fsync_file(model_temp)
        model_size = model_temp.stat().st_size
        model_sha256 = _sha256_file(model_temp)
        replay_manifest = _capture_replay_manifest(model.replay_buffer, sidecar_dir)
        manifest = {
            "version": CHECKPOINT_VERSION,
            "created_unix": time.time(),
            "env_step": int(model.env_steps_exact),
            "policy_step": int(model.num_timesteps),
            "config_signature": config_signature(cfg),
            "model_state": {"entry": MODEL_STATE_ENTRY, "size": model_size, "sha256": model_sha256},
            "replay_sidecar": sidecar_dir.name,
            "replay": replay_manifest,
        }
        with zipfile.ZipFile(
            checkpoint_temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            archive.write(model_temp, MODEL_STATE_ENTRY)
            archive.writestr(
                MANIFEST_ENTRY,
                json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=_json_value),
            )
        _fsync_file(checkpoint_temp)
        _validate_checkpoint_file(checkpoint_temp, model_size, model_sha256)
        os.replace(checkpoint_temp, path)
        _fsync_directory(path.parent)
        return manifest
    finally:
        if model_temp.exists():
            model_temp.unlink()
        if checkpoint_temp.exists():
            checkpoint_temp.unlink()


def load_training_checkpoint(
    path,
    model,
    logging_callback,
    numbered_checkpoint_callback,
    latest_checkpoint_callback,
    cfg,
    allow_config_mismatch=False,
):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        manifest = json.loads(archive.read(MANIFEST_ENTRY))
        if manifest["version"] != CHECKPOINT_VERSION:
            raise ResumeCheckpointError(
                f"Unsupported checkpoint version: {manifest['version']} (expected {CHECKPOINT_VERSION})"
            )
        current_signature = config_signature(cfg)
        if manifest["config_signature"] != current_signature and not allow_config_mismatch:
            changed = {
                key: {
                    "checkpoint": manifest["config_signature"].get(key),
                    "current": current_signature.get(key),
                }
                for key in sorted(set(manifest["config_signature"]) | set(current_signature))
                if manifest["config_signature"].get(key) != current_signature.get(key)
            }
            raise ResumeCheckpointError(
                f"Resume config mismatch: {json.dumps(changed, sort_keys=True)}"
            )
        model_bytes = io.BytesIO(archive.read(MODEL_STATE_ENTRY))
        if hashlib.sha256(model_bytes.getbuffer()).hexdigest() != manifest["model_state"]["sha256"]:
            raise ResumeCheckpointError("Model state checksum mismatch")
        model_state = _torch_load(model_bytes, map_location=model.device)

    _restore_model_state(model, model_state)
    sidecar_dir = path.parent / manifest["replay_sidecar"]
    _restore_replay_buffer(model.replay_buffer, manifest["replay"], sidecar_dir)
    restore_logging_callback_state(logging_callback, model_state["logging_callback"])
    numbered_state = model_state["numbered_checkpoint_callback"]
    numbered_checkpoint_callback.n_calls = int(numbered_state["n_calls"])
    numbered_checkpoint_callback.num_timesteps = int(numbered_state["num_timesteps"])
    latest_checkpoint_callback.load_state_dict(model_state["latest_checkpoint_callback"])
    latest_checkpoint_callback.set_pending_rng_state(model_state["rng"])
    return model_state["training"], manifest


class AtomicLatestCheckpointCallback(BaseCallback):
    def __init__(
        self,
        path,
        logging_callback,
        numbered_checkpoint_callback,
        cfg,
        training_state,
        interval_env_steps=DEFAULT_INTERVAL_ENV_STEPS,
        retry_seconds=60,
        verbose=1,
    ):
        super().__init__(verbose=verbose)
        self.path = str(path)
        self.logging_callback = logging_callback
        self.numbered_checkpoint_callback = numbered_checkpoint_callback
        self.cfg = cfg
        self.training_state = training_state
        self.interval_env_steps = int(interval_env_steps)
        if self.interval_env_steps <= 0:
            raise ValueError("latest checkpoint interval_env_steps must be positive")
        self.next_checkpoint_env_step = self.interval_env_steps
        self.pending_save = False
        self.retry_seconds = int(retry_seconds)
        self.next_retry_monotonic = 0.0
        self.pending_rng_state = None

    def state_dict(self, saved_env_step=None):
        next_step = self.next_checkpoint_env_step
        pending = self.pending_save
        if saved_env_step is not None:
            while next_step <= int(saved_env_step):
                next_step += self.interval_env_steps
            pending = False
        return {
            "n_calls": int(self.n_calls),
            "num_timesteps": int(self.num_timesteps),
            "next_checkpoint_env_step": int(next_step),
            "pending_save": bool(pending),
        }

    def load_state_dict(self, state):
        self.n_calls = int(state["n_calls"])
        self.num_timesteps = int(state["num_timesteps"])
        self.next_checkpoint_env_step = int(state["next_checkpoint_env_step"])
        self.pending_save = bool(state.get("pending_save", False))

    def set_pending_rng_state(self, rng_state):
        self.pending_rng_state = rng_state

    def _on_training_start(self):
        if self.pending_rng_state is not None:
            restore_rng_state(self.pending_rng_state)
            self.pending_rng_state = None
            print("[resume] restored Python/NumPy/Torch RNG state", flush=True)

    def _on_step(self):
        env_step = int(getattr(self.model, "env_steps_exact", 0))
        if env_step >= self.next_checkpoint_env_step:
            self.pending_save = True
        return True

    def _on_rollout_start(self):
        if self.pending_save:
            self._save_if_due()

    def _on_training_end(self):
        if self.pending_save:
            self._save_if_due()

    def _save_if_due(self):
        if time.monotonic() < self.next_retry_monotonic:
            return
        env_step = int(self.model.env_steps_exact)
        latest_state = self.state_dict(saved_env_step=env_step)
        try:
            manifest = save_training_checkpoint(
                self.path,
                self.model,
                self.logging_callback,
                self.numbered_checkpoint_callback,
                latest_state,
                self.training_state,
                self.cfg,
            )
        except Exception as exc:
            self.next_retry_monotonic = time.monotonic() + self.retry_seconds
            print(
                f"[latest-checkpoint-error] old checkpoint preserved path={self.path} "
                f"env_step={env_step} retry_seconds={self.retry_seconds} error={exc!r}",
                flush=True,
            )
            return
        self.load_state_dict(latest_state)
        self.next_retry_monotonic = 0.0
        print(
            f"[latest-checkpoint] saved path={self.path} env_step={manifest['env_step']} "
            f"policy_step={manifest['policy_step']} replay_rows={manifest['replay']['size']}",
            flush=True,
        )
