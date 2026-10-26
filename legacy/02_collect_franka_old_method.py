"""Two Franka Panda arms fold a cloth in half; scripted demos recorded to disk."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from franka_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--out", default=str(ROOT / "dataset_raw"))
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-orientation", action="store_true")
ap.add_argument("--max-tries", type=int, default=6)
ap.add_argument("--probe", action="store_true")
ap.add_argument("--policy-port", type=int, default=0)
ap.add_argument("--grab-gate", type=float, default=0.025)
ap.add_argument("--max-drift", type=float, default=0.06)
ap.add_argument("--double-fold", action="store_true")
ap.add_argument("--line-pin", action="store_true")
ap.add_argument("--edge-tol", type=float, default=0.04)
ap.add_argument("--place-z", type=float, default=0.02)
ap.add_argument("--place-slow", type=float, default=1.0)
ap.add_argument("--dwell", type=float, default=0.0)
args = ap.parse_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import torch
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Vt
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim, ClothPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, interface_config_loader
from isaacsim.storage.native import get_assets_root_path
from omni.physx.scripts import particleUtils, physicsUtils
import isaacsim.core.utils.numpy.rotations as rot_utils
try:
    from isaacsim.storage.native import is_file
except Exception:
    is_file = None

rng = np.random.default_rng(args.seed)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
conn = None
ROLL_N = 0
LAST_POSE = None
if args.policy_port:
    from multiprocessing.connection import Client
    conn = Client(("localhost", args.policy_port), authkey=b"wam")


def P_(*a):
    print(*a, flush=True)

def _T(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device="cuda")

def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.asarray(x)


class BaseIK:
    """Lula IK in world frame + forward kinematics for the real gripper pose."""
    def __init__(self, lula, cfg):
        self.lula = lula
        self.bpos = np.array(cfg["pos"], dtype=np.float64)
        yaw = np.deg2rad(cfg["yaw_deg"])
        c, s = np.cos(yaw), np.sin(yaw)
        self.Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        quat = np.asarray(euler_angles_to_quat(np.array([0, 0, yaw])), dtype=np.float64)
        lula.set_robot_base_pose(self.bpos, quat)
        self.fk_is_world = True
        self.warm = None

    def calibrate(self):
        tgt = self.bpos + self.Rz @ np.array([0.45, 0.0, 0.30])
        q, ok = self.lula.compute_inverse_kinematics(EE_FRAME, tgt, None)
        if not ok:
            P_("[ik] calibration IK failed"); return
        p, _ = self.lula.compute_forward_kinematics(EE_FRAME, np.asarray(q, dtype=np.float64))
        self.fk_is_world = bool(np.linalg.norm(np.asarray(p) - tgt) < 0.02)
        P_(f"[ik] fk_is_world={self.fk_is_world} (fk={np.round(p, 3)} target={np.round(tgt, 3)})")

    def fk(self, q):
        p, _ = self.lula.compute_forward_kinematics(EE_FRAME, np.asarray(q, dtype=np.float64))
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return p if self.fk_is_world else self.bpos + self.Rz @ p

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        q, ok = self.lula.compute_inverse_kinematics(
            EE_FRAME, np.asarray(target_position, dtype=np.float64), target_orientation,
            warm_start=None if self.warm is None else np.asarray(self.warm, dtype=np.float64))
        class _A: pass
        act = _A(); act.joint_positions = np.asarray(q, dtype=np.float64)
        if ok:
            self.warm = act.joint_positions.reshape(-1)[:7].copy()
        return act, ok


# ---------------------------------------------------------------- scene
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / CONTROL_HZ,
              backend="torch", device="cuda")
stage = omni.usd.get_context().get_stage()
scene_path = "/physicsScene"
if not stage.GetPrimAtPath(scene_path):
    UsdPhysics.Scene.Define(stage, scene_path)

root = get_assets_root_path()
try:
    add_reference_to_stage(root + "/Isaac/Environments/Grid/default_environment.usd", "/World/Environment")
except Exception as e:
    P_("[scene] no NVIDIA environment, plain ground:", e)
    world.scene.add_default_ground_plane()

cands = [root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", root + "/Isaac/Robots/Franka/franka.usd"]
FRANKA_USD = cands[0]
if is_file is not None:
    for c in cands:
        try:
            if is_file(c):
                FRANKA_USD = c; break
        except Exception:
            pass
P_("[scene] franka usd:", FRANKA_USD)

FixedCuboid(prim_path="/World/Table", name="table",
            position=np.array([0.0, 0.0, TABLE_Z / 2]),
            scale=np.array([TABLE_SIZE[0], TABLE_SIZE[1], TABLE_Z]),
            color=np.array([0.55, 0.42, 0.3]))

ARMS = {}
for side, cfg in ROBOT_BASES.items():
    path = f"/World/Franka_{side}"
    add_reference_to_stage(FRANKA_USD, path)
    xf = SingleXFormPrim(path)
    xf.set_world_pose(position=np.array(cfg["pos"]),
                      orientation=euler_angles_to_quat(np.array([0, 0, np.deg2rad(cfg["yaw_deg"])])))
    art_path = path
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_path = str(p.GetPath()); break
    for p in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    ARMS[side] = dict(path=path, art=SingleArticulation(prim_path=art_path, name=f"franka_{side}"))

# ---- cloth (PhysX PBD particle cloth)
CLOTH_PATH = "/World/Cloth"; PS_PATH = "/World/ParticleSystem"
n = CLOTH_RES
lin = np.linspace(-CLOTH_SIZE / 2, CLOTH_SIZE / 2, n)
gx, gy = np.meshgrid(lin, lin)
REST = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], 1)
faces, counts = [], []
for j in range(n - 1):
    for i in range(n - 1):
        a = j * n + i
        faces += [a, a + 1, a + n + 1, a + n]; counts.append(4)
mesh = UsdGeom.Mesh.Define(stage, CLOTH_PATH)
mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in REST + np.array([*CLOTH_CENTER, TABLE_Z + 0.01])]))
mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces))
mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
mesh.GetDoubleSidedAttr().Set(True)
mesh.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
particleUtils.add_physx_particle_system(
    stage=stage, particle_system_path=PS_PATH, simulation_owner=scene_path,
    contact_offset=0.008, rest_offset=0.0035, particle_contact_offset=0.008,
    solid_rest_offset=0.0035, fluid_rest_offset=0.0,
    solver_position_iterations=16, global_self_collision_enabled=True,
    non_particle_collision_enabled=True)
particleUtils.add_pbd_particle_material(stage, "/World/ClothMaterial", friction=0.9, damping=0.05)
physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(PS_PATH), "/World/ClothMaterial")
particleUtils.add_physx_particle_cloth(
    stage=stage, path=CLOTH_PATH, dynamic_mesh_path=None, particle_system_path=PS_PATH,
    spring_stretch_stiffness=1.0e4, spring_bend_stiffness=50.0, spring_shear_stiffness=500.0,
    spring_damping=0.5, self_collision=True, self_collision_filter=True)

# ---- cameras
def make_cam(name, pos, euler_deg):
    cam = Camera(prim_path=f"/World/cam_{name}", resolution=(IMG_RES, IMG_RES), frequency=CONTROL_HZ)
    cam.set_world_pose(np.array(pos), rot_utils.euler_angles_to_quats(np.array(euler_deg), degrees=True), camera_axes="world")
    return cam
CAMS = {
    "overhead": make_cam("overhead", (0.0, -1.0, 0.9), (0, 34, 90)),
    "front":    make_cam("front", (0.0, 0.95, 0.80), (0, 30, -90)),
}

world.reset()
for c in CAMS.values():
    c.initialize()
cloth = ClothPrim(prim_paths_expr=CLOTH_PATH, name="cloth")
cloth.initialize()

# ---- kinematics
lula_cfg = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
for side, a in ARMS.items():
    art = a["art"]; art.initialize()
    names = list(art.dof_names)
    P_(f"[dof] {side}: {names}")
    a["arm_idx"] = np.array([names.index(j) for j in ARM_JOINTS])
    a["fin_idx"] = np.array([names.index(j) for j in FINGER_JOINTS])
    kps = np.full(len(names), 2000.0); kds = np.full(len(names), 200.0)
    kps[a["fin_idx"]] = 2000.0; kds[a["fin_idx"]] = 100.0
    art.get_articulation_controller().set_gains(kps=_T(kps), kds=_T(kds))
    a["ik"] = BaseIK(LulaKinematicsSolver(**lula_cfg), ROBOT_BASES[side])
    a["ik"].calibrate()

TOOL_DOWN = np.array([0.0, 1.0, 0.0, 0.0])    # (w,x,y,z): tool z-axis points at the table
ORIENT = None if args.no_orientation else TOOL_DOWN


def hold(a, arm_q, grip):
    """Set the drive TARGET so the arm holds this pose."""
    q = np.zeros(len(a["art"].dof_names))
    q[a["arm_idx"]] = arm_q; q[a["fin_idx"]] = grip
    a["art"].apply_action(ArticulationAction(joint_positions=_T(q)))


def teleport(a, arm_q, grip):
    q = np.zeros(len(a["art"].dof_names))
    q[a["arm_idx"]] = arm_q; q[a["fin_idx"]] = grip
    a["art"].set_joint_positions(_T(q))
    a["art"].set_joint_velocities(_T(np.zeros_like(q)))
    hold(a, arm_q, grip)                      # <- target matches pose, no swinging
    a["last_arm_cmd"] = np.asarray(arm_q, dtype=np.float64).copy()


def actual_ee(side):
    a = ARMS[side]
    q = _np(a["art"].get_joint_positions()).reshape(-1)[a["arm_idx"]].astype(np.float64)
    return a["ik"].fk(q)


# ---------------------------------------------------------------- cloth helpers
def cloth_world(yaw, center_xy):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return REST @ R.T + np.array([center_xy[0], center_xy[1], TABLE_Z + 0.008])

def get_pts():
    return _np(cloth.get_world_positions()).reshape(-1, 3).copy()

def set_pts(p, pinned=None):
    t = torch.as_tensor(p.reshape(1, -1, 3), dtype=torch.float32, device="cuda")
    v = None
    if pinned is not None:
        try:
            v = torch.as_tensor(cloth.get_velocities()).clone().reshape(1, -1, 3).to("cuda")
        except Exception:
            v = None
    cloth.set_world_positions(t)
    try:
        if v is None:
            cloth.set_velocities(torch.zeros_like(t))
        else:
            v[:, torch.as_tensor(pinned, dtype=torch.long, device="cuda"), :] = 0.0
            cloth.set_velocities(v)
    except Exception:
        pass


class Pins:
    """Kinematic grasp: while closed, a 3x3 cloth patch follows the REAL gripper pose."""
    def __init__(self):
        self.idx = None; self.off = None
    def grab(self, idx, ee_pos):
        self.idx = idx; self.off = get_pts()[idx] - ee_pos
    def release(self):
        self.idx = None
    def apply(self, ee_pos, pts):
        if self.idx is not None:
            pts[self.idx] = ee_pos + self.off

def patch(i0, j0):
    return np.array([(j0 + dj) * n + (i0 + di) for dj in range(3) for di in range(3)])


def step_physics():
    for k in range(SUBSTEPS):
        world.step(render=(k == SUBSTEPS - 1))

def grab_images():
    out = {}
    for name, cam in CAMS.items():
        rgba = cam.get_rgba()
        out[name] = np.zeros((IMG_RES, IMG_RES, 3), np.uint8) if rgba is None or np.size(rgba) == 0 else _np(rgba)[..., :3].astype(np.uint8)
    return out

def get_state():
    s = []
    for side in ("left", "right"):
        a = ARMS[side]; q = _np(a["art"].get_joint_positions()).reshape(-1)
        s += list(q[a["arm_idx"]]) + [float(q[a["fin_idx"]].mean())]
    return np.array(s, np.float32)

def smooth(a):
    return 10 * a ** 3 - 15 * a ** 4 + 6 * a ** 5


class Episode:
    def __init__(self):
        self.state, self.action, self.frames = [], [], {k: [] for k in CAMS}
        self.ik_fail = 0; self.ik_total = 0; self.miss = False
        self.phase = 0; self.pdrift = []; self.track = {}; self.fallback = {'left': 0, 'right': 0}
        self.grip = {"left": GRIP_OPEN, "right": GRIP_OPEN}

    def control_step(self, targets, grip_targets, pins, record=True):
        if record:
            self.state.append(get_state())
            for k, v in grab_images().items():
                self.frames[k].append(v)
        act = []
        for side in ("left", "right"):
            a = ARMS[side]
            _e = float(np.linalg.norm(np.asarray(targets[side]) - actual_ee(side)))
            self.track[(self.phase, side)] = max(self.track.get((self.phase, side), 0.0), _e)
            self.ik_total += 1
            action, ok = a["ik"].compute_inverse_kinematics(targets[side], ORIENT)
            if not ok and ORIENT is not None:
                self.fallback[side] += 1
                action, ok = a["ik"].compute_inverse_kinematics(targets[side])
            if ok:
                a["last_arm_cmd"] = np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)[:7]
            else:
                self.ik_fail += 1
            self.grip[side] += np.clip(grip_targets[side] - self.grip[side], -0.01, 0.01)
            hold(a, a["last_arm_cmd"], self.grip[side])
            act += list(a["last_arm_cmd"]) + [self.grip[side]]
        if record:
            self.action.append(np.array(act, np.float32))
        if any(p.idx is not None for p in pins.values()):
            pts = get_pts()
            for side in ("left", "right"):
                pins[side].apply(actual_ee(side), pts)
            set_pts(pts, np.concatenate([pn.idx for pn in pins.values() if pn.idx is not None]))
        step_physics()


def rollout(ep, world_pts, Pn, T=190):
    global ROLL_N
    ROLL_N += 1
    Path("/tmp/rollout").mkdir(exist_ok=True)
    conn.send(("reset",)); conn.recv()
    pins = {s: Pins() for s in ARMS}
    grabbed = {s: False for s in ARMS}; released = {s: False for s in ARMS}
    mingap = {s: 9.9 for s in ARMS}; mingrip = {s: 9.9 for s in ARMS}
    minreach = {s: 9.9 for s in ARMS}; close_gap = {s: None for s in ARMS}
    close_t = {s: None for s in ARMS}; rel_err = {s: None for s in ARMS}
    _fmt = lambda v: "none" if v is None else f"{v:.3f}"
    for t in range(T):
        imgs = grab_images(); st = get_state()
        if ROLL_N <= 3 and t % 24 == 0:
            for cam, im in imgs.items():
                Image.fromarray(im).save(f"/tmp/rollout/try{ROLL_N}_{cam}_{t:03d}.jpg")
        conn.send(("act", {"overhead": (imgs["overhead"].tobytes(), imgs["overhead"].shape),
                           "front": (imgs["front"].tobytes(), imgs["front"].shape),
                           "state": [float(x) for x in st]}))
        a16 = np.asarray(conn.recv(), dtype=np.float64)
        for k, side in enumerate(("left", "right")):
            a = ARMS[side]
            q = a16[k * 8:k * 8 + 7]; g = float(np.clip(a16[k * 8 + 7], 0.0, GRIP_OPEN))
            hold(a, q, g)
            gs = float(st[k * 8 + 7])
            mingrip[side] = min(mingrip[side], g)
            _pn = get_pts()
            gap_now = float(np.linalg.norm(_pn[Pn[side]["idx"]].mean(0) - actual_ee(side)))
            minreach[side] = min(minreach[side], gap_now)
            if g < 0.006 and close_gap[side] is None:
                close_gap[side] = gap_now; close_t[side] = t
            if pins[side].idx is None and not released[side] and gs < 0.006 and g < 0.006:
                ee = actual_ee(side)
                gp = float(np.linalg.norm(get_pts()[Pn[side]["idx"]].mean(0) - ee))
                mingap[side] = min(mingap[side], gp)
                if gp < args.grab_gate:
                    pins[side].grab(Pn[side]["idx"], ee); grabbed[side] = True
            elif pins[side].idx is not None and g > 0.02:
                pins[side].release(); released[side] = True
                rel_err[side] = float(np.mean([np.linalg.norm(_pn[c] - world_pts[(n - 1) * n + c]) for c in (0, n // 2, n - 1)]))
        if any(p.idx is not None for p in pins.values()):
            pts = get_pts()
            for side in ("left", "right"):
                pins[side].apply(actual_ee(side), pts)
            set_pts(pts, np.concatenate([pn.idx for pn in pins.values() if pn.idx is not None]))
        step_physics()
    for _ in range(15):
        step_physics()
    final = get_pts(); cols = [0, n // 2, n - 1]
    err = float(np.mean([np.linalg.norm(final[c] - world_pts[(n - 1) * n + c]) for c in cols]))
    drift = float(np.mean([np.linalg.norm(final[(n - 1) * n + c] - world_pts[(n - 1) * n + c]) for c in cols]))
    ok = bool(np.isfinite(final).all() and err < 0.04 and drift < 0.06
              and final[:, 2].max() < TABLE_Z + 0.10 and all(grabbed.values()))
    _p0, _p1 = world_pts[0], world_pts[n - 1]
    yaw0 = float(np.degrees(np.arctan2(_p1[1] - _p0[1], _p1[0] - _p0[0])))
    cx0, cy0 = world_pts[:, :2].mean(0)
    P_(f"[rollout] {ROLL_N}: edge_err={err:.3f} drift={drift:.3f} grabbed={grabbed} released={released}"
       f" | yaw={yaw0:+.1f} c=({cx0:+.3f},{cy0:+.3f}) mingap L/R={mingap['left']:.3f}/{mingap['right']:.3f} mingrip L/R={mingrip['left']:.3f}/{mingrip['right']:.3f}"
       f" reach L/R={minreach['left']:.3f}/{minreach['right']:.3f} close_gap L/R={_fmt(close_gap['left'])}/{_fmt(close_gap['right'])}"
       f" close_t L/R={close_t['left']}/{close_t['right']} rel_err L/R={_fmt(rel_err['left'])}/{_fmt(rel_err['right'])}")
    return ep, ok, err


def run_episode(ep_id):
    yaw = np.deg2rad(rng.uniform(-15, 15))
    center = np.array(CLOTH_CENTER) + rng.uniform(-0.03, 0.03, 2)
    world_pts = cloth_world(yaw, center)
    global LAST_POSE
    LAST_POSE = (float(np.degrees(yaw)), float(center[0]), float(center[1]))
    P_(f"[pose] yaw={LAST_POSE[0]:+.1f} cx={LAST_POSE[1]:+.3f} cy={LAST_POSE[2]:+.3f}")
    set_pts(world_pts)
    Pn = {}
    spec = {"left": (0, 0, 0, n - 1), "right": (n - 1, 0, n - 1, n - 1)}
    for side, (i0, j0, i1, j1) in spec.items():
        if args.line_pin:
            idx = np.arange(0, n // 2) if i0 == 0 else np.arange(n // 2, n)   # half of the near edge row
            ref = 1 if i0 == 0 else n - 2                                       # hand holds near the corner
            src = world_pts[ref].copy()
            Pn[side] = dict(src=src, dst=src + (world_pts[j1 * n + i1] - world_pts[j0 * n + i0]), idx=idx, ref=ref)
        else:
            idx = patch(0 if i0 == 0 else n - 3, 0)
            src = world_pts[idx].mean(0)
            Pn[side] = dict(src=src, dst=src + (world_pts[j1 * n + i1] - world_pts[j0 * n + i0]), idx=idx)
    start = {s: Pn[s]["src"] + np.array([0, -0.10, 0.20]) for s in Pn}
    for side in ARMS:
        a = ARMS[side]
        a["ik"].warm = HOME_Q.copy()
        teleport(a, HOME_Q, GRIP_OPEN)
        for _ in range(6):
            action, ok = a["ik"].compute_inverse_kinematics(start[side], ORIENT)
        if ok:
            teleport(a, np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)[:7], GRIP_OPEN)
    UsdGeom.Mesh(stage.GetPrimAtPath(CLOTH_PATH)).CreateDisplayColorAttr([Gf.Vec3f(*rng.uniform(0.1, 0.9, 3))])
    for _ in range(int(1.0 * CONTROL_HZ)):
        step_physics()
    set_pts(world_pts)                              # re-seat the cloth after settling

    ep = Episode()
    if args.policy_port:
        return rollout(ep, world_pts, Pn)
    z_grasp = TABLE_Z + 0.012
    jit = lambda: rng.normal(0, 0.004, 3)
    wp = []
    def add(dur, pos, grip, pin=None):
        wp.append((dur * rng.uniform(0.85, 1.15), {s: pos[s] + jit() * (pin is None) for s in pos}, grip, pin))
    add(1.5, {s: Pn[s]["src"] + [0, 0, 0.12] for s in Pn}, GRIP_OPEN)
    add(1.0, {s: Pn[s]["src"] + [0, 0, z_grasp - Pn[s]["src"][2]] for s in Pn}, GRIP_OPEN)
    add(0.6, {s: Pn[s]["src"] + [0, 0, z_grasp - Pn[s]["src"][2]] for s in Pn}, GRIP_CLOSED, "grab")
    add(1.2, {s: Pn[s]["src"] + 0.35 * (Pn[s]["dst"] - Pn[s]["src"]) + [0, 0, 0.10] for s in Pn}, GRIP_CLOSED)
    add(1.8, {s: Pn[s]["dst"] + [0, 0, 0.12] for s in Pn}, GRIP_CLOSED)
    add(1.0 * args.place_slow, {s: Pn[s]["dst"] + [0, 0, args.place_z] for s in Pn}, GRIP_CLOSED)
    add(1.0 * args.place_slow, {s: Pn[s]["dst"] + [0, 0, args.place_z] for s in Pn}, GRIP_OPEN, "release")
    add(1.0, {s: Pn[s]["dst"] + [0, 0, 0.15] for s in Pn}, GRIP_OPEN)

    pins = {s: Pins() for s in ARMS}
    cur = {s: start[s].copy() for s in ARMS}
    PG = Pn
    _rt = []
    def _aln(second):
        q = get_pts()
        if second:
            rr = np.arange(n // 2 + 1, n); v = q[rr * n, :2] - q[rr * n + n - 1, :2]
        else:
            v = q[0:n, :2] - q[(n - 1) * n:(n - 1) * n + n, :2]
        m = v.mean(0)
        dz = (q[rr * n, 2] - q[rr * n + n - 1, 2]).mean() if second else (q[0:n, 2] - q[(n - 1) * n:(n - 1) * n + n, 2]).mean()
        ex = ""
        if not second and hasattr(ep, "far_ref"):
            fd = (q[(n - 1) * n:(n - 1) * n + n, :2] - ep.far_ref).mean(0)
            nd = q[0:n, :2].mean(0) - (actual_ee("left")[:2] + actual_ee("right")[:2]) / 2 - ep.near_off
            ex = f" F({fd[0]:+.3f},{fd[1]:+.3f}) N({nd[0]:+.3f},{nd[1]:+.3f})"
        return f"{np.linalg.norm(v, axis=1).mean():.3f}({m[0]:+.3f},{m[1]:+.3f}) dz{dz:+.3f}" + ex
    for ph, (dur, tgt, grip, pin) in enumerate(wp):
        ep.phase = ph
        T = max(2, int(dur * CONTROL_HZ))
        begin = {s: cur[s].copy() for s in ARMS}
        for k in range(1, T + 1):
            al = smooth(k / T)
            pos = {s: begin[s] + al * (tgt[s] - begin[s]) for s in ARMS}
            if pin == "grab" and k == T:
                for s in PG:
                    ee = actual_ee(s)
                    gap = float(np.linalg.norm((get_pts()[PG[s]["ref"]] if "ref" in PG[s] else get_pts()[PG[s]["idx"]].mean(0)) - ee))
                    track = float(np.linalg.norm(pos[s] - ee))
                    P_(f"[grasp] {s}: gripper-to-cloth gap={gap:.3f}m  target-vs-actual={track:.3f}m  ee={np.round(ee,3)}")
                    if gap < 0.025:
                        pins[s].grab(PG[s]["idx"], ee)
                    else:
                        ep.miss = True
            if pin == "release" and k == T:
                _g = get_pts(); ep.relvec = getattr(ep, "relvec", None) if ph >= 8 else np.mean([_g[c] - world_pts[(n - 1) * n + c] for c in (0, n // 2, n - 1)], axis=0)
                for s in ARMS:
                    pins[s].release()
            ep.control_step(pos, {s: grip for s in ARMS}, pins)
            if args.line_pin and ((ph in (5, 13) and k == T) or (ph in (6, 14) and k % 3 == 0) or (ph in (7, 15) and k % 3 == 0)):
                _rt.append(f"p{ph}k{k}:" + _aln(ph >= 8))
        cur = {s: tgt[s].copy() for s in ARMS}
        if ph in (6, 14) and args.dwell > 0:
            for _ in range(int(args.dwell * CONTROL_HZ)):
                ep.control_step(cur, {s: GRIP_OPEN for s in ARMS}, pins)
        _f = get_pts()
        if args.line_pin:
            if not hasattr(ep, "align"):
                ep.align = []
            if ph < 8:
                _v = _f[0:n, :2] - _f[(n - 1) * n:(n - 1) * n + n, :2]
            else:
                _rr = np.arange(n // 2 + 1, n)
                _v = _f[_rr * n, :2] - _f[_rr * n + n - 1, :2]
            ep.align.append((float(np.linalg.norm(_v, axis=1).mean()), _v.mean(0)))
        if ph < 8:
            ep.pdrift.append(float(np.mean([np.linalg.norm(_f[(n - 1) * n + c] - world_pts[(n - 1) * n + c]) for c in (0, n // 2, n - 1)])))
        if args.line_pin and ph == 4:        # aim the carried edge at the LIVE far edge before lowering
            _c = get_pts()
            ep.far_ref = _c[(n - 1) * n:(n - 1) * n + n, :2].copy()
            ep.near_off = _c[0:n, :2].mean(0) - (actual_ee("left")[:2] + actual_ee("right")[:2]) / 2
            for sd in Pn:
                _ix = Pn[sd]["idx"]
                _d = (_c[(n - 1) * n + _ix].mean(0) - _c[_ix].mean(0))[:2]
                _goal = actual_ee(sd)[:2] + _d
                for kk in (5, 6, 7):
                    _du, _tg, _gr, _pn = wp[kk]
                    _tg = dict(_tg); _tg[sd] = np.array([_goal[0], _goal[1], _tg[sd][2]])
                    wp[kk] = (_du, _tg, _gr, _pn)
        if args.double_fold and ph == 7:
            _h = get_pts()
            ep.snap1 = _h.copy()
            ep.err1 = float(np.mean([np.linalg.norm(_h[c] - world_pts[(n - 1) * n + c]) for c in (0, n // 2, n - 1)]))
            if not ep.miss:
                idx2 = np.array([j * n for j in range(n)]) if args.line_pin else np.array([j * n + i for j in range(n) for i in range(3)])
                src2 = _h[idx2].mean(0)
                dst2 = src2 + (world_pts[n - 1] - world_pts[0])
                PG = {"left": dict(src=src2, dst=dst2, idx=idx2)}
                park = start["right"]
                def add2(dur, lpos, grip, pin=None):
                    add(dur, {"left": lpos, "right": park}, grip, pin)
                zg = np.array([0, 0, z_grasp - src2[2]])
                add2(2.0, src2 + [0, 0, 0.12], GRIP_OPEN)
                add2(1.0, src2 + zg, GRIP_OPEN)
                add2(0.6, src2 + zg, GRIP_CLOSED, "grab")
                add2(1.2, src2 + 0.35 * (dst2 - src2) + [0, 0, 0.12], GRIP_CLOSED)
                add2(2.2, dst2 + [0, 0, 0.14], GRIP_CLOSED)
                add2(1.0 * args.place_slow, dst2 + [0, 0, args.place_z + 0.01], GRIP_CLOSED)
                add2(1.0 * args.place_slow, dst2 + [0, 0, args.place_z + 0.01], GRIP_OPEN, "release")
                add2(1.0, dst2 + [0, 0, 0.15], GRIP_OPEN)
        if args.double_fold and args.line_pin and ph == 12 and "left" in PG:
            _c = get_pts()
            _rw = np.arange(n)
            _d = (_c[_rw * n + (n - 1)] - _c[_rw * n]).mean(0)[:2]
            _goal = actual_ee("left")[:2] + _d
            for kk in (13, 14, 15):
                _du, _tg, _gr, _pn = wp[kk]
                _tg = dict(_tg); _tg["left"] = np.array([_goal[0], _goal[1], _tg["left"][2]])
                wp[kk] = (_du, _tg, _gr, _pn)
    _hold = int(2.0 * CONTROL_HZ)
    _tr = []
    for _k in range(_hold):
        ep.control_step(cur, {s: GRIP_OPEN for s in ARMS}, pins)
        if args.line_pin and (_k + 1) % max(1, _hold // 5) == 0:
            _tr.append(_aln(len(wp) > 8))
    if args.line_pin:
        P_("[release] " + " ".join(_rt))
        P_("[settle] every 0.4s after retreat: " + " ".join(_tr))

    final = get_pts()
    cols = [0, n // 2, n - 1]
    err = float(np.mean([np.linalg.norm(final[0 * n + c] - world_pts[(n - 1) * n + c]) for c in cols]))
    _s = ep.snap1 if (args.double_fold and hasattr(ep, "snap1")) else final
    _e = np.array([np.linalg.norm(_s[c] - world_pts[(n - 1) * n + c]) for c in range(n)])
    _r = np.array([np.linalg.norm(_s[c][:2] - _s[(n - 1) * n + c][:2]) for c in range(n)])
    P_(f"[edge] fold1 vs original far edge mean={_e.mean():.3f} max={_e.max():.3f} | vs actual far edge (xy) mean={_r.mean():.3f} max={_r.max():.3f}")
    if args.line_pin:
        err = float(_r.mean())
        P_("[align] " + " ".join(f"p{k}:{a:.3f}({v[0]:+.3f},{v[1]:+.3f})" for k, (a, v) in enumerate(getattr(ep, "align", []))))
    P_("[fold] near edge vs ORIGINAL far edge (left, mid, right): " + ", ".join(f"{np.linalg.norm(final[c] - world_pts[(n-1)*n + c]):.3f}" for c in cols))
    P_("[fold] far edge drift from start (left, mid, right):      " + ", ".join(f"{np.linalg.norm(final[(n-1)*n + c] - world_pts[(n-1)*n + c]):.3f}" for c in cols))
    for sd in ("left", "right"):
        P_(f"[track] {sd}: " + " ".join(f"p{ph}={ep.track.get((ph, sd), 0.0):.3f}" for ph in range(8)) + f"  orient_fallbacks={ep.fallback[sd]}")
    _rv = getattr(ep, "relvec", None)
    _fv = np.mean([final[c] - world_pts[(n - 1) * n + c] for c in cols], axis=0)
    P_("[fold-vec] release: " + ("n/a" if _rv is None else np.array2string(_rv, precision=3)) + "  final: " + np.array2string(_fv, precision=3))
    P_("[phase-drift] far-edge drift after p0..p7: " + " ".join(f"{d:.3f}" for d in ep.pdrift))
    if args.double_fold:
        if args.line_pin:
            _rows = range(n // 2 + 1, n)
            _e2 = np.array([np.linalg.norm(final[j * n][:2] - final[j * n + n - 1][:2]) for j in _rows])
        else:
            _rows = (n // 2 + 2, (3 * n) // 4, n - 1)
            _e2 = np.array([np.linalg.norm(final[j * n] - world_pts[j * n + n - 1]) for j in _rows])
        err2 = float(_e2.mean())
        err1 = float(_r.mean()) if (args.line_pin and hasattr(ep, "snap1")) else getattr(ep, "err1", 9.9)
        P_(f"[fold2] first-fold err={err1:.3f}  left-edge vs right-edge err={err2:.3f} max={_e2.max():.3f}")
        err = max(err1, err2)
    drift = float(np.mean([np.linalg.norm(_s[(n - 1) * n + c] - world_pts[(n - 1) * n + c]) for c in cols]))
    ok = bool(np.isfinite(final).all() and err < args.edge_tol and final[:, 2].max() < TABLE_Z + 0.10
              and ep.ik_fail / max(1, ep.ik_total) < 0.05 and not ep.miss and drift < args.max_drift)
    return ep, ok, float(err)


def save_episode(ep, k):
    d = OUT / f"ep_{k:06d}"
    for cam in CAMS:
        (d / cam).mkdir(parents=True, exist_ok=True)
        for t, img in enumerate(ep.frames[cam]):
            Image.fromarray(img).save(d / cam / f"{t:06d}.jpg", quality=92)
    np.savez(d / "data.npz", state=np.stack(ep.state), action=np.stack(ep.action))
    (d / "meta.json").write_text(json.dumps(dict(task=TASK_TEXT, fps=CONTROL_HZ, length=len(ep.state), pose=LAST_POSE)))


if args.probe:
    def _q2R(q):
        x, y, z, w = [float(v) for v in q]
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    for _ in range(12):
        step_physics()
    cx, cy = CLOTH_CENTER
    variants = [("z1.0_p90", (cx, cy, 1.0), (0, 90, 0)), ("z1.0_pm90", (cx, cy, 1.0), (0, -90, 0)),
                ("z0.8_p90", (cx, cy, 0.8), (0, 90, 0)), ("z1.2_p90", (cx, cy, 1.2), (0, 90, 0)),
                ("z1.0_p90_yaw90", (cx, cy, 1.0), (0, 90, 90)), ("tilt_near", (0.0, -1.0, 0.9), (0, 40, 90)),
                ("f_p20", (0.0, 0.95, 0.80), (0, 20, -90)), ("f_p30", (0.0, 0.95, 0.80), (0, 30, -90)),
                ("f_close_p30", (0.0, 0.70, 0.70), (0, 30, -90)), ("f_close_p40", (0.0, 0.70, 0.70), (0, 40, -90))]
    for i, (nm, pos, eul) in enumerate(variants):
        c = Camera(prim_path=f"/World/sweep_{i}", resolution=(IMG_RES, IMG_RES), frequency=CONTROL_HZ)
        c.set_world_pose(np.array(pos), rot_utils.euler_angles_to_quats(np.array(eul), degrees=True), camera_axes="world")
        c.initialize()
        for _ in range(12):
            step_physics()
        r = c.get_rgba()
        if r is None or np.size(r) == 0:
            P_(f"[sweep] {nm}: EMPTY"); continue
        r = _np(r)[..., :3]
        P_(f"[sweep] {nm}: mean_rgb={r.mean():.1f}")
        Image.fromarray(r.astype(np.uint8)).save(f"/tmp/sweep_{nm}.png")
    for name, cam in CAMS.items():
        rgba = cam.get_rgba()
        arr = None if rgba is None or np.size(rgba) == 0 else _np(rgba)
        if arr is None:
            P_(f"[probe] cam {name}: EMPTY frame")
        else:
            P_(f"[probe] cam {name}: shape={arr.shape} mean_rgb={arr[..., :3].mean():.1f}")
            Image.fromarray(arr[..., :3].astype(np.uint8)).save(f"/tmp/probe_{name}.png")
    tg = {"left": np.array([-0.15, -0.2, TABLE_Z + 0.15]), "right": np.array([0.15, -0.2, TABLE_Z + 0.15])}
    for side, a in ARMS.items():
        a["ik"].warm = HOME_Q.copy()
        teleport(a, HOME_Q, GRIP_OPEN)
        for _ in range(6):
            action, ok = a["ik"].compute_inverse_kinematics(tg[side], ORIENT)
        teleport(a, np.asarray(action.joint_positions, dtype=np.float64).reshape(-1)[:7], GRIP_OPEN)
    for _ in range(20):
        step_physics()
    for side, a in ARMS.items():
        fk = actual_ee(side)
        try:
            v = a["art"]._articulation_view._physics_view
            names = list(v.shared_metatype.link_names)
            T = _np(v.get_link_transforms())[0]
            h = names.index("panda_hand")
            tcp = T[h, :3] + _q2R(T[h, 3:7]) @ np.array([0, 0, 0.1034])
            P_(f"[probe] {side}: target={np.round(tg[side],3)} lula_fk={np.round(fk,3)} sim_hand_tcp={np.round(tcp,3)} diff={np.linalg.norm(fk - tcp):.3f}m")
        except Exception as e:
            P_(f"[probe] {side}: link pose read failed: {type(e).__name__}: {e}")
    app.close(); sys.exit(0)

saved = len(list(OUT.glob("ep_*")))
tried = 0; t0 = time.time()
target = saved + args.episodes
while saved < target and tried < args.max_tries and app.is_running():
    tried += 1
    ep, ok, err = run_episode(saved)
    P_(f"[ep] try {tried}: ok={ok} corner_err={err:.3f}m ik_fail={ep.ik_fail}/{ep.ik_total} frames={len(ep.state)}")
    if args.policy_port:
        saved += int(ok)
    elif ok:
        save_episode(ep, saved); saved += 1
P_(f"[done] saved {saved - (target - args.episodes)} / {tried} tries in {time.time() - t0:.0f}s -> {OUT}")
app.close()
