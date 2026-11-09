# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TCP pickle RPC server for frozen Cosmos LIBERO q50 policy inference."""

import argparse
import os
import pickle
import socketserver
import struct
import time
import traceback
from typing import Dict

import numpy as np

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Serve Cosmos LIBERO q50 actions for ExRL")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host for local AAC RPC clients.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19057,
        help="Bind port for local AAC RPC clients.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cosmos_predict2_2b_480p_libero90_task57_q50__inference_only",
        help="Cosmos inference experiment config.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Cosmos checkpoint path, usually .../iter_000001500/model.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="cosmos_policy/config/config.py",
        help="Cosmos config file path.",
    )
    parser.add_argument(
        "--dataset_stats_path",
        type=str,
        required=True,
        help="Cosmos dataset_statistics.json for proprio/action scaling.",
    )
    parser.add_argument(
        "--t5_text_embeddings_path",
        type=str,
        required=True,
        help="Cosmos t5_embeddings.pkl containing task language embeddings.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50,
        help="Predicted action horizon.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=195,
        help="Deterministic Cosmos sampling seed.",
    )
    parser.add_argument(
        "--randomize_seed",
        action="store_true",
        help="Randomize Cosmos action sampling seed per query.",
    )
    parser.add_argument(
        "--num_denoising_steps_action",
        type=int,
        default=5,
        help="Cosmos action denoising steps.",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_90",
        help="Cosmos/LIBERO normalization key.",
    )
    return parser.parse_args()


_HEADER = struct.Struct("!Q")


def _resolve_ckpt_path(path: str) -> str:
    path = str(path).rstrip("/")
    if os.path.basename(path) == "model" and os.path.exists(os.path.join(path, ".metadata")):
        return os.path.dirname(path)
    return path


def _recvall(sock, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("client closed socket")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock, payload: Dict) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def _recv_message(sock) -> Dict:
    header = _recvall(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    return pickle.loads(_recvall(sock, size))


class CosmosAACPolicy:
    def __init__(self, args):
        self.args = args
        resolved_ckpt_path = _resolve_ckpt_path(args.ckpt_path)
        self.cfg = PolicyEvalConfig(
            suite="libero",
            config=args.config,
            ckpt_path=resolved_ckpt_path,
            config_file=args.config_file,
            use_third_person_image=True,
            use_wrist_image=True,
            use_proprio=True,
            flip_images=False,
            use_variance_scale=False,
            use_jpeg_compression=True,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            num_denoising_steps_action=int(args.num_denoising_steps_action),
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            unnormalize_actions=True,
            normalize_proprio=True,
            dataset_stats_path=args.dataset_stats_path,
            t5_text_embeddings_path=args.t5_text_embeddings_path,
            trained_with_image_aug=True,
            chunk_size=int(args.chunk_size),
            num_open_loop_steps=int(args.chunk_size),
            deterministic=True,
            task_suite_name=args.task_suite_name,
            task_id=57,
            seed=int(args.seed),
            randomize_seed=bool(args.randomize_seed),
            data_collection=False,
        )
        print("Loading Cosmos dataset stats...", flush=True)
        self.dataset_stats = load_dataset_stats(args.dataset_stats_path)
        print("Initializing T5 embeddings...", flush=True)
        init_t5_text_embeddings_cache(args.t5_text_embeddings_path)
        print("Loading Cosmos policy checkpoint...", flush=True)
        self.model, _ = get_model(self.cfg)
        self.metadata = {
            "model_family": "cosmos",
            "ckpt_path": args.ckpt_path,
            "resolved_ckpt_path": resolved_ckpt_path,
            "config": args.config,
            "chunk_size": int(args.chunk_size),
            "action_dim": 7,
            "dataset_stats_path": args.dataset_stats_path,
            "t5_text_embeddings_path": args.t5_text_embeddings_path,
            "num_denoising_steps_action": int(args.num_denoising_steps_action),
            "randomize_seed": bool(args.randomize_seed),
            "supports_infer_batch": True,
        }
        print(f"Cosmos policy server ready: {self.metadata}", flush=True)

    def infer(self, obs: Dict) -> Dict:
        start = time.time()
        task = str(obs.get("prompt") or obs.get("task") or obs.get("task_description"))
        policy_obs = {
            "primary_image": np.asarray(obs["primary_image"], dtype=np.uint8),
            "wrist_image": np.asarray(obs["wrist_image"], dtype=np.uint8),
            "proprio": np.asarray(obs["proprio"], dtype=np.float32),
        }
        result = get_action(
            self.cfg,
            self.model,
            self.dataset_stats,
            policy_obs,
            task,
            seed=int(self.args.seed),
            randomize_seed=bool(self.args.randomize_seed),
            num_denoising_steps_action=int(self.args.num_denoising_steps_action),
            generate_future_state_and_value_in_parallel=False,
        )
        infer_sec = time.time() - start
        actions = np.asarray(result["actions"], dtype=np.float32)
        return {
            "actions_bytes": actions.tobytes(),
            "actions_shape": tuple(actions.shape),
            "actions_dtype": str(actions.dtype),
            "policy_timing": {
                "infer_sec": float(infer_sec),
                "infer_ms": float(infer_sec * 1000.0),
            },
        }

    def infer_batch(self, observations, padded_batch_size: int = 0) -> Dict:
        start = time.time()
        observations = list(observations)
        actual_batch_size = len(observations)
        if actual_batch_size <= 0:
            raise ValueError("infer_batch requires at least one observation")
        outputs = [self.infer(obs) for obs in observations]
        actions = np.stack(
            [
                np.frombuffer(item["actions_bytes"], dtype=item["actions_dtype"]).reshape(
                    item["actions_shape"]
                )
                for item in outputs
            ]
        )
        padded_batch_size = actual_batch_size
        infer_sec = time.time() - start
        return {
            "actions_batch_bytes": actions.tobytes(),
            "actions_batch_shape": tuple(actions.shape),
            "actions_batch_dtype": str(actions.dtype),
            "policy_timing": {
                "infer_sec": float(infer_sec),
                "infer_ms": float(infer_sec * 1000.0),
                "batch_size": int(actual_batch_size),
                "padded_batch_size": int(padded_batch_size),
                "infer_sec_per_observation": float(infer_sec / actual_batch_size),
            },
        }


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            request = _recv_message(self.request)
            method = request.get("method")
            if method == "metadata":
                response = self.server.policy.metadata
            elif method == "infer":
                response = self.server.policy.infer(request["obs"])
            elif method == "infer_batch":
                response = self.server.policy.infer_batch(
                    request["observations"],
                    padded_batch_size=int(request.get("padded_batch_size", 0)),
                )
            else:
                response = {"error": f"Unknown method: {method}"}
        except Exception:
            response = {"error": traceback.format_exc()}
        _send_message(self.request, response)


class _Server(socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler, policy):
        super().__init__(server_address, handler)
        self.policy = policy


def main():
    args = parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            "The pickle RPC server must bind to loopback; use SSH forwarding for remote access"
        )
    policy = CosmosAACPolicy(args)
    with _Server((args.host, int(args.port)), _Handler, policy) as server:
        print(f"Serving Cosmos AAC policy on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
