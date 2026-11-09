"""Launch H=50 Cosmos base-policy training in the Cosmos environment."""

import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def run(recipe, args):
    if args.print_config:
        print(yaml.safe_dump(recipe, sort_keys=False))
        return
    if args.norm_stats or args.resume:
        raise ValueError(
            "Cosmos computes dataset statistics while loading; use a fresh output directory"
        )
    data = Path(args.data).expanduser().resolve()
    checkpoint = Path(args.base_policy or "").expanduser().resolve()
    tokenizer = Path(recipe["tokenizer_checkpoint"]).expanduser().resolve()
    for path in (data / "t5_embeddings.pkl", checkpoint, tokenizer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not any(data.glob("*.hdf5")):
        raise FileNotFoundError(f"No regenerated LIBERO demonstrations in {data}")
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    native = dict(
        recipe,
        dataset=str(data),
        checkpoint=str(checkpoint),
        output=str(output),
        tokenizer_checkpoint=str(tokenizer),
    )
    settings = output / "pretraining.json"
    settings.write_text(json.dumps(native, indent=2))
    env = dict(
        os.environ,
        EXRL_COSMOS_RECIPE=str(settings),
        WANDB_MODE="disabled",
        IMAGINAIRE_OUTPUT_ROOT=str(output),
        COSMOS_POLICY_SKIP_HF_RESOLVE="true",
        COSMOS_POLICY_BASE_CKPT_PATH=str(checkpoint),
        COSMOS_POLICY_TOKENIZER_CKPT_PATH=str(tokenizer),
    )
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={recipe['training']['gpus']}",
        "-m",
        "cosmos_policy.scripts.train",
        "--config=exrl/pretraining/cosmos_config.py",
        "--",
        "experiment=exrl_libero_h50",
    ]
    subprocess.run(command, env=env, check=True)
