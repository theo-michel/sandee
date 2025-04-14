# put this in your lerebot folder
python ../lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Drive to the red block and pick it up" \
  --control.repo_id=theo-michel/eval_act_lekiwi_testv_$((RANDOM % 1000000000)) \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=0 \
  --control.episode_time_s=30 \
  --control.reset_time_s=5 \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_lekiwi_testv2/checkpoints/last/pretrained_model

  # --control.policy.path=/Users/theomichel/Coding/Hackaton/sandee/act_lekiwi_v10/checkpoints/080000/pretrained_model