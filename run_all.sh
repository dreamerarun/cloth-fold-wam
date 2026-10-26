#!/usr/bin/env bash
# One script for the whole pipeline. Save as /home/arun/wam/run_all.sh
#
#   bash /home/arun/wam/run_all.sh                 # everything: build, collect 200 eps, convert, train ACT
#   EPISODES=20 bash run_all.sh                    # smaller run
#   POLICY=smolvla bash run_all.sh                 # train SmolVLA instead of ACT
#   bash run_all.sh build|test|collect|convert|train   # run just one stage
#
# Stop at any stage with Ctrl+C; re-running skips work that is already done.
set -euo pipefail

WAM=/home/arun/wam
ISAAC=/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
CONDA_ENV=lerobot
EPISODES=${EPISODES:-200}
POLICY=${POLICY:-act}                      # act | smolvla
STAGE=${1:-all}

say() { echo; echo "=================== $* ==================="; }
want() { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

# make `conda activate` usable inside this script
CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f "$CONDA_SH" ] || CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"

cd "$ISAAC"

if want build; then
  say "1/5  Build WAM (download model, make URDF + USD)"
  ./python.sh -m pip install -q xacro
  if [ -f "$WAM/assets/wam7_gripper.usd" ]; then
    echo "USD already exists, skipping (delete $WAM/assets to rebuild)"
  else
    ./python.sh "$WAM/01_build_wam.py"
  fi
fi

if want test; then
  say "2/5  Smoke test: 2 episodes, with GUI (close the window or wait for it to finish)"
  ./python.sh "$WAM/02_collect_franka.py" --episodes 2 --out "$WAM/dataset_test"
fi

if want collect; then
  say "3/5  Collect $EPISODES episodes (headless)"
  have=$(ls -d "$WAM"/dataset_raw/ep_* 2>/dev/null | wc -l || true)
  if [ "$have" -ge "$EPISODES" ]; then
    echo "already have $have episodes, skipping"
  else
    ./python.sh "$WAM/02_collect_franka.py" --episodes $((EPISODES - have)) --max-tries $(( (EPISODES - have) * 5 )) --seed $have --headless --out "$WAM/dataset_raw"
  fi
fi

if want convert; then
  say "4/5  Convert to LeRobot dataset (conda env: $CONDA_ENV)"
  source "$CONDA_SH"; conda activate "$CONDA_ENV"
  python "$WAM/03_convert_to_lerobot.py" --raw "$WAM/dataset_raw" \
         --repo-id local/wam_fold --root "$HOME/lerobot_data/wam_fold"
fi

if want train; then
  say "5/5  Train $POLICY (conda env: $CONDA_ENV)"
  source "$CONDA_SH"; conda activate "$CONDA_ENV"
  cd "$WAM"
  bash "$WAM/04_train.sh" "$POLICY"
fi

say "finished"
