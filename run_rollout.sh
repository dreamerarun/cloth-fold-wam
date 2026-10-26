#!/usr/bin/env bash
# usage: bash run_rollout.sh <tries> <seed> <logfile> [extra args for 02_collect_franka.py]
TRIES=$1; SEED=$2; LOG=$3; shift 3
ISAAC=/home/arun/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
CKPT=/home/arun/wam/outputs/train/wam_fold_act/checkpoints/last/pretrained_model
( source /home/arun/miniconda3/etc/profile.d/conda.sh; conda activate lerobot; cd /home/arun/wam
  exec python policy_server.py "$CKPT" --port 6000 --device cpu > /tmp/policy_server.log 2>&1 ) &
SRV=$!
for i in $(seq 1 60); do grep -q "waiting on port" /tmp/policy_server.log 2>/dev/null && break; sleep 2; done
grep -q "waiting on port" /tmp/policy_server.log || { echo "server did not start:"; tail -n 20 /tmp/policy_server.log; kill $SRV; exit 1; }
cd "$ISAAC"
PYTHONUNBUFFERED=1 ./python.sh /home/arun/wam/02_collect_franka.py --headless --policy-port 6000 --episodes 1000 --max-tries "$TRIES" --seed "$SEED" --out /tmp/rollout_out_$SEED "$@" > "$LOG" 2>&1
kill $SRV 2>/dev/null
grep -a "^\[done\]" "$LOG"
