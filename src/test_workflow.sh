

lerobot-eval --policy.path=outputs/cfg_1025/checkpoints/020000/pretrained_model --policy.n_action_steps=10 --policy.chunk_size=30 --env.type=libero --env.task=libero_10 --eval.batch_size=5 --eval.n_episodes=5 --env.max_parallel_tasks=2 --output_dir=./eval_result/cfg_1025_default/chunk_30_action_5