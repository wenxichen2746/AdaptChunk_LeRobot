#!/usr/bin/env python

from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# POLICY_PATH = Path("outputs/default_smolvla_1102/checkpoints/015000/pretrained_model")
# OUTPUT_ROOT = Path("./eval_result/default_smolvla_1102")

POLICY_PATH = Path("outputs/default_smolvla1109/checkpoints/025000/pretrained_model")
OUTPUT_ROOT = Path("./eval_result/default_smolvla1109_25k")

EVAL_SEED = 1000
EVAL_SETTINGS = SimpleNamespace(batch_size=5, n_episodes=30, use_async_envs=False, max_episodes_rendered=3)
ENV_SETTINGS = LiberoEnv(task="libero_10", max_parallel_tasks=1)
# EXECUTION_HORIZONS = [3, 5, 7, 9, 11, 13, 15]
CHUNK_SIZES = [1, 5, 10, 20, 30, 40, 50]


def build_policy_config(
    decoding_strategy: str,
    decoding_kwargs: dict,
    n_action_steps: int,
    chunk_size: int,
) -> PreTrainedConfig:
    overrides = [
        f"--n_action_steps={n_action_steps}",
        f"--chunk_size={chunk_size}",
        f"--decoding_strategy={decoding_strategy}",
    ]
    policy_cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=POLICY_PATH,
        cli_overrides=overrides,
    )
    policy_cfg.pretrained_path = str(POLICY_PATH)
    policy_cfg.decoding_kwargs = copy.deepcopy(decoding_kwargs)
    policy_cfg.n_action_steps = n_action_steps
    policy_cfg.chunk_size = chunk_size
    return policy_cfg


def run_eval(
    run_name: str,
    decoding_strategy: str,
    decoding_kwargs: dict,
    n_action_steps: int,
    chunk_size: int,
) -> dict:
    policy_cfg = build_policy_config(decoding_strategy, decoding_kwargs, n_action_steps, chunk_size)
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    if EVAL_SETTINGS.max_episodes_rendered and EVAL_SETTINGS.max_episodes_rendered > 0:
        videos_dir.mkdir(parents=True, exist_ok=True)
    else:
        videos_dir = None

    device = get_safe_torch_device(policy_cfg.device, log=True)
    envs = make_env(ENV_SETTINGS, n_envs=EVAL_SETTINGS.batch_size, use_async_envs=EVAL_SETTINGS.use_async_envs)

    policy = make_policy(cfg=policy_cfg, env_cfg=ENV_SETTINGS)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if policy_cfg.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=EVAL_SETTINGS.n_episodes,
            max_episodes_rendered=EVAL_SETTINGS.max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=EVAL_SEED,
            max_parallel_tasks=ENV_SETTINGS.max_parallel_tasks,
        )

    close_envs(envs)
    torch.save(info, output_dir / "results.pt")
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def main():
    init_logging()
    set_seed(EVAL_SEED)
    evaluations = []
    for chunk_size in CHUNK_SIZES:
        # for execution_horizon in EXECUTION_HORIZONS:
        for execution_horizon in [chunk_size]:
            prefix = f"c{chunk_size}_h{execution_horizon}"
            evaluations.append((f"{prefix}_naive", "naive", {}, execution_horizon, chunk_size))

    start_time = time.time()
    elapsed_history: list[float] = []

    for idx, (run_name, strategy, kwargs, n_action_steps, chunk_size) in enumerate(evaluations, start=1):
        print(
            f"Running {run_name} with strategy='{strategy}', "
            f"kwargs={kwargs}, n_action_steps={n_action_steps}, chunk_size={chunk_size}"
        )
        run_start = time.time()
        info = run_eval(run_name, strategy, kwargs, n_action_steps, chunk_size)
        run_elapsed = time.time() - run_start
        elapsed_history.append(run_elapsed)

        avg_elapsed = sum(elapsed_history) / len(elapsed_history)
        remaining_runs = len(evaluations) - idx
        eta_seconds = remaining_runs * avg_elapsed

        print(f"==== {run_name} ====")
        # print(info["overall"])
        print(
            f"Run time: {run_elapsed/60:.2f} min | "
            f"Average: {avg_elapsed/60:.2f} min | "
            f"ETA for remaining {remaining_runs} runs: {eta_seconds/60/60:.2f} hours"
        )

    total_elapsed = time.time() - start_time
    print(f"All evaluations completed in {total_elapsed/60/60:.2f} hours.")

if __name__ == "__main__":
    main()
