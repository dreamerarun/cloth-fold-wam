# Bimanual cloth folding in Isaac Sim 5.1.0

Double fold of a cloth with two Franka arms, with kinematic particle pins on the held edge.

## Methods (same script, different flags)

**Old method** - lift to 10-14 cm, arc over, lower, open, retreat. Pins applied once per control step.

    ./python.sh 02_collect_franka.py --line-pin --double-fold --place-z 0.012

**New method** - low lift (~4 cm), rolling descent, pins re-applied every physics substep.

    ./python.sh 02_collect_franka.py --line-pin --double-fold --pin-substeps --drag-fold --place-z 0.012

`legacy/02_collect_franka_old_method.py` is a snapshot of the script from before `--pin-substeps` and `--drag-fold` existed.

## Other flags
`--bend`, `--place-z2`, `--jit`, `--damp` tune cloth stiffness, fold-2 release height, waypoint noise and damping (defaults keep the original behavior).

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `01_build_wam.py` | Builds the two-arm WAM scene in Isaac Sim |
| 2 | `02_collect_franka.py` | Collects cloth-folding demonstrations (flags above) |
| 3 | `03_convert_to_lerobot.py` | Converts raw demos to the LeRobot dataset format |
| 4 | `04_train.sh` | Trains the policy |
| - | `policy_server.py`, `run_rollout.sh` | Serves a trained policy and runs rollouts in sim |
| - | `run_all.sh` | Runs the steps end to end |
| - | `wam_common.py`, `franka_common.py` | Shared helpers |

Datasets (`dataset_*/`), `assets/` and `outputs/` are not in the repo (see `.gitignore`).
