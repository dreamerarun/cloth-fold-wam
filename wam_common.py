"""Shared constants. No Isaac Sim imports here so every script can use it."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.resolve()
ASSETS = ROOT / "assets"
REPO_DIR = ASSETS / "barrett_model"
URDF_PATH = ASSETS / "wam7_gripper.urdf"
USD_PATH = ASSETS / "wam7_gripper.usd"
LULA_YAML = ASSETS / "wam7_lula.yaml"

# ---- robot ---------------------------------------------------------------
# canonical WAM joint order j1..j7 (names produced by 01_build_wam.py)
ARM_JOINTS = [
    "wam_base_yaw_joint",        # j1
    "wam_shoulder_pitch_joint",  # j2
    "wam_shoulder_yaw_joint",    # j3 (upper-arm roll)
    "wam_elbow_pitch_joint",     # j4
    "wam_wrist_yaw_joint",       # j5
    "wam_wrist_pitch_joint",     # j6
    "wam_palm_yaw_joint",        # j7
]
FINGER_JOINTS = ["wam_finger_l_joint", "wam_finger_r_joint"]
EE_FRAME = "wam_tool_link"           # fingertip frame added by 01_build_wam.py
BASE_LINK = "wam_base_link"

# arm leans forward, elbow bent, hand pointing down. Eyeball it with --gui
# on the first run and tweak if the arm looks odd.
HOME_Q = np.array([0.0, 0.5, 0.0, 2.0, 0.0, 0.6, 0.0])

GRIP_OPEN = 0.045    # metres per finger
GRIP_CLOSED = 0.0

# ---- scene ---------------------------------------------------------------
TABLE_Z = 0.25                       # table top height (m)
TABLE_SIZE = (1.2, 0.9)
# two arms side by side on the same long edge, both facing +y
ROBOT_BASES = {
    "left":  dict(pos=(-0.30, -0.50, 0.0), yaw_deg=90.0),
    "right": dict(pos=(+0.30, -0.50, 0.0), yaw_deg=90.0),
}
CLOTH_SIZE = 0.30                    # m, square
CLOTH_RES = 30                       # particles per side (30x30 = 900)
CLOTH_CENTER = (0.0, -0.02)          # xy of cloth centre on the table

# ---- recording -----------------------------------------------------------
PHYSICS_HZ = 120
CONTROL_HZ = 15
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ
IMG_RES = 224
TASK_TEXT = "fold the cloth in half"
