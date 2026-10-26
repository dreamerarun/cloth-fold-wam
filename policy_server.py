"""Run in the lerobot env:  python policy_server.py <pretrained_model_dir> [--port 6000] [--device cpu]"""
import argparse, numpy as np, torch
from multiprocessing.connection import Listener
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--port", type=int, default=6000); ap.add_argument("--device", default="cpu")
a = ap.parse_args()

policy = ACTPolicy.from_pretrained(a.ckpt); policy.to(a.device); policy.config.device = a.device; policy.eval()
try:
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
except TypeError:
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt)
print("[server] policy loaded; waiting on port", a.port, flush=True)

def to_dev(b):
    return {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in b.items()}

def img(t):
    buf, shape = t
    x = np.frombuffer(buf, dtype=np.uint8).reshape(shape)[..., :3].copy()
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

with Listener(("localhost", a.port), authkey=b"wam") as L:
    conn = L.accept()
    print("[server] client connected", flush=True)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg[0] == "reset":
            policy.reset()
            for p in (pre, post):
                if hasattr(p, "reset"):
                    p.reset()
            conn.send("ok")
        else:
            d = msg[1]
            batch = {"observation.state": torch.tensor(d["state"], dtype=torch.float32).unsqueeze(0),
                     "observation.images.overhead": img(d["overhead"]),
                     "observation.images.front": img(d["front"])}
            batch = to_dev(pre(batch))
            with torch.no_grad():
                act = post(policy.select_action(batch))
            conn.send(act.detach().cpu().numpy().reshape(-1).astype(float).tolist())
