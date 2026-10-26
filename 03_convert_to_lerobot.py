"""Convert dataset_raw/ep_*/ into a LeRobotDataset. Run in your existing `lerobot` conda env:

    conda activate lerobot
    python 03_convert_to_lerobot.py --raw dataset_raw --repo-id local/wam_fold --root ~/lerobot_data/wam_fold
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset            # newer lerobot
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset     # older lerobot

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="dataset_raw")
ap.add_argument("--repo-id", default="local/wam_fold")
ap.add_argument("--root", default=str(Path.home() / "lerobot_data" / "wam_fold"))
a = ap.parse_args()

eps = sorted(Path(a.raw).glob("ep_*"))
assert eps, f"no episodes in {a.raw}"
meta0 = json.loads((eps[0] / "meta.json").read_text())
cams = sorted(p.name for p in eps[0].iterdir() if p.is_dir())
res = np.asarray(Image.open(next((eps[0] / cams[0]).glob("*.jpg")))).shape      # H,W,3

names16 = [f"{s}_{j}" for s in ("left", "right") for j in ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]]
features = {
    "observation.state": {"dtype": "float32", "shape": (16,), "names": names16},
    "action": {"dtype": "float32", "shape": (16,), "names": names16},
}
for c in cams:
    features[f"observation.images.{c}"] = {"dtype": "video", "shape": res, "names": ["height", "width", "channels"]}

if Path(a.root).exists():
    shutil.rmtree(a.root)
ds = LeRobotDataset.create(repo_id=a.repo_id, fps=meta0["fps"], features=features,
                           root=a.root, robot_type="dual_wam", use_videos=True)

for e in eps:
    d = np.load(e / "data.npz")
    meta = json.loads((e / "meta.json").read_text())
    for t in range(meta["length"]):
        frame = {"observation.state": d["state"][t], "action": d["action"][t]}
        for c in cams:
            frame[f"observation.images.{c}"] = np.asarray(Image.open(e / c / f"{t:06d}.jpg"))
        try:
            ds.add_frame({**frame, "task": meta["task"]})
        except TypeError:
            ds.add_frame(frame, task=meta["task"])
    ds.save_episode()
    print("converted", e.name)

if hasattr(ds, "consolidate"):      # only exists in older lerobot versions
    ds.consolidate()
if hasattr(ds, "finalize"):          # newer lerobot: flushes meta/episodes parquet
    ds.finalize()
print("done ->", a.root)
