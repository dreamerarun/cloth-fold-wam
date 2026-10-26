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
