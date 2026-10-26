"""Dual Franka Panda constants. Reuses recording constants from wam_common."""
import numpy as np
from wam_common import ROOT, PHYSICS_HZ, CONTROL_HZ, SUBSTEPS, IMG_RES, TASK_TEXT, CLOTH_SIZE, CLOTH_RES

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
EE_FRAME = "right_gripper"                      # fingertip frame in Isaac's Franka Lula config
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])   # hand pointing down
GRIP_OPEN = 0.04
GRIP_CLOSED = 0.0

TABLE_Z = 0.25
TABLE_SIZE = (1.4, 1.2)
ROBOT_BASES = {                                  # mounted on the table top, facing +y
    "left":  dict(pos=(-0.30, -0.52, TABLE_Z), yaw_deg=90.0),
    "right": dict(pos=(+0.30, -0.52, TABLE_Z), yaw_deg=90.0),
}
CLOTH_CENTER = (0.0, -0.05)
