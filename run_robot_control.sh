#!/bin/bash

# Source conda instead of initializing
# source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path to your conda installation
# conda activate lerobot
source venv/bin/activate

# Run the script directly instead of with 'python'
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Drive to the red block and pick it up" \
  --control.repo_id=theo-michel/eval_act_lekiwi_testv_70 \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_lekiwi_testv2/checkpoints/last/pretrained_model