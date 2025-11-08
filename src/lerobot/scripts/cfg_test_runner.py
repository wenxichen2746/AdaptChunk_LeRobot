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

# POLICY_PATH = Path("outputs/cfg_1025/checkpoints/020000/pretrained_model")
# OUTPUT_ROOT = Path("./eval_result/tests_cfg_1025_chunksize50")
# POLICY_PATH = Path("outputs/batch64_actioncondition_nocfg/checkpoints/020000/pretrained_model")
# OUTPUT_ROOT = Path("./eval_result/tests_nocfg_chunksize50")
# POLICY_PATH = Path("outputs/cfg_1030_drpastactions_40k/checkpoints/040000/pretrained_model")
# OUTPUT_ROOT = Path("./eval_result/cfg_1030_drpastactions_40k_1102")
POLICY_PATH = Path("outputs/cfg_1104_20pastactions/checkpoints/040000/pretrained_model")
OUTPUT_ROOT = Path("./eval_result/cfg_1104_20pastactions")

EVAL_SEED = 1000
EVAL_SETTINGS = SimpleNamespace(batch_size=5, n_episodes=10,use_async_envs=False, max_episodes_rendered=3)
ENV_SETTINGS = LiberoEnv(task="libero_10", max_parallel_tasks=1)


def build_policy_config(decoding_strategy: str, decoding_kwargs: dict, n_action_steps: int) -> PreTrainedConfig:
    overrides = [
        f"--n_action_steps={n_action_steps}",
        "--chunk_size=50",
        f"--decoding_strategy={decoding_strategy}",
    ]
    policy_cfg = PreTrainedConfig.from_pretrained(
        pretrained_name_or_path=POLICY_PATH,
        cli_overrides=overrides,
    )
    policy_cfg.pretrained_path = str(POLICY_PATH)
    policy_cfg.decoding_kwargs = copy.deepcopy(decoding_kwargs)
    policy_cfg.n_action_steps = n_action_steps
    return policy_cfg


def run_eval(
    run_name: str,
    decoding_strategy: str,
    decoding_kwargs: dict,
    n_action_steps: int,
) -> dict:
    policy_cfg = build_policy_config(decoding_strategy, decoding_kwargs, n_action_steps)
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
    for execution_horizon in [10,15,20,1,5,30]:
        # print(f"=== Evaluation with execution_horizon={execution_horizon} ===")
        prefix = f"h{execution_horizon}"
        evaluations.append((f"{prefix}_naive_nulla", "naive_nulla", {}, execution_horizon))
        evaluations.append((f"{prefix}_naive", "naive", {}, execution_horizon))
        evaluations.append(
            (
                f"{prefix}_rtc_exp",
                "rtc",
                {"prefix_attention_schedule": "EXP"},
                execution_horizon,
            )
        )
        evaluations.append(
            (
                f"{prefix}_bid_exp",
                "bid",
                {"prefix_attention_schedule": "exp"},
                execution_horizon,
            )
        )


        for w in [1.2,1.4,1.6]:
            w_nn = 1 - w -1
            w_on = w
            w_ao = 0.0
            w_na = 1.0
            evaluations.append(
                (
                    f"{prefix}_cfg_BI_wo_{w}",
                    "cfg",
                    {"w_ao": w_ao, "w_on": w_on, "w_na": w_na, "w_nn": w_nn},
                    execution_horizon,
                )
            )
        for w in [1.2,1.4,1.6]:
            w_nn = 1 - w -1
            w_on = 1.0 #updated
            w_ao = 0.0
            w_na = w
            evaluations.append(
                (
                    f"{prefix}_cfg_BI_wa_{w}",
                    "cfg",
                    {"w_ao": w_ao, "w_on": w_on, "w_na": w_na, "w_nn": w_nn},
                    execution_horizon,
                )
            )
        for w in [1, 1.2,1.4,1.6]:
            w_nn = 0.0
            w_on = 1 - w
            w_ao = w
            w_na = 0.0
            evaluations.append(
                (
                    f"{prefix}_cfg_BF_wa_{w}",
                    "cfg",
                    {"w_ao": w_ao, "w_on": w_on, "w_na": w_na, "w_nn": w_nn},
                    execution_horizon,
                )
            )

    start_time = time.time()
    elapsed_history: list[float] = []

    for idx, (run_name, strategy, kwargs, n_action_steps) in enumerate(evaluations, start=1):
        print(
            f"Running {run_name} with strategy='{strategy}', "
            f"kwargs={kwargs}, n_action_steps={n_action_steps}"
        )
        run_start = time.time()
        info = run_eval(run_name, strategy, kwargs, n_action_steps)
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
