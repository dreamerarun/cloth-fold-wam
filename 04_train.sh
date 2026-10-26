#!/usr/bin/env bash
# Run inside your lerobot conda env.  Usage:  bash 04_train.sh act   |   bash 04_train.sh smolvla
# Flag names change between lerobot versions. If one is rejected, run `lerobot-train --help`.
set -e
REPO=local/wam_fold
ROOT=$HOME/lerobot_data/wam_fold

if [ "$1" = "act" ]; then
  lerobot-train \
    --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.video_backend=pyav \
    --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
    --policy.chunk_size=30 --policy.n_action_steps=30 \
    --batch_size=8 --steps=60000 --save_freq=10000 --num_workers=4 \
    --output_dir=outputs/train/wam_fold_act --job_name=wam_fold_act --wandb.enable=false

elif [ "$1" = "smolvla" ]; then
  # smolvla_base was trained with 3 cameras named camera1..3; map our 2 cameras and pad one.
  lerobot-train \
    --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.overhead":"observation.images.camera1","observation.images.front":"observation.images.camera2"}' \
    --dataset.repo_id=$REPO --dataset.root=$ROOT --dataset.video_backend=pyav \
    --batch_size=2 --steps=30000 --save_freq=5000 --num_workers=2 \
    --output_dir=outputs/train/wam_fold_smolvla --job_name=wam_fold_smolvla --wandb.enable=false
else
  echo "usage: bash 04_train.sh act|smolvla"; exit 1
fi
