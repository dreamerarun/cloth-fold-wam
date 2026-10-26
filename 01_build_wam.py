"""Download the Barrett WAM model from the web and turn it into a USD for Isaac Sim.

Run with Isaac Sim's own python:
    cd ~/Downloads/isaac-sim-standalone-5.1.0-linux-x86_64
    ./python.sh -m pip install xacro
    ./python.sh /path/to/wam_cloth/01_build_wam.py

Add --urdf-only to stop after the URDF (no Isaac Sim needed, any python with xacro).
"""
import argparse, re, shutil, subprocess, sys, urllib.request, zipfile, io
from xml.dom import minidom
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wam_common import *

REPO_URL = "https://github.com/tu-rbo/barrett_model.git"
ZIP_URL = "https://codeload.github.com/tu-rbo/barrett_model/zip/HEAD"


def download_repo():
    if (REPO_DIR / "models").exists():
        print("[wam] repo already present")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    except Exception as e:  # no git? fall back to the zip
        print("[wam] git failed (%s), downloading zip" % e)
        data = urllib.request.urlopen(ZIP_URL).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            top = z.namelist()[0].split("/")[0]
            z.extractall(ASSETS)
        shutil.move(str(ASSETS / top), str(REPO_DIR))


def expand_xacro():
    import xacro
    # the xacro files use ROS $(find ..) / package:// - point them at the local copy
    work = ASSETS / "_xacro_work"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(REPO_DIR, work, ignore=shutil.ignore_patterns(".git"))
    for f in list(work.glob("models/*.xacro")) + list(work.glob("robots/*.xacro")):
        t = f.read_text()
        t = t.replace("$(find barrett_model)", str(work)).replace("package://barrett_model", str(work))
        f.write_text(t)

    doc = xacro.process_file(str(work / "robots" / "wam7.urdf.xacro"))
    root = doc.documentElement

    # leftover macro definitions / comments from the legacy xacro namespace
    for c in list(root.childNodes):
        if c.nodeType == 8 or (c.nodeType == 1 and "${" in (c.getAttribute("name") or "")):
            root.removeChild(c)

    def strip(n):
        for c in list(n.childNodes):
            if c.nodeType == 8:
                n.removeChild(c)
            elif c.nodeType == 1:
                if "macro" in c.tagName or c.tagName.startswith("xacro:"):
                    n.removeChild(c)
                else:
                    strip(c)
    strip(root)

    # drop the 'world' link and the fixed joint that lifts the arm 1 m (we place it ourselves)
    for c in list(root.childNodes):
        if c.nodeType != 1:
            continue
        if c.tagName == "link" and c.getAttribute("name") == "world":
            root.removeChild(c)
        if c.tagName == "joint" and c.getAttribute("name") == "wam/wam_fixed_joint":
            root.removeChild(c)
    return root.toxml()


def sanitize(xml: str) -> str:
    # 'wam/shoulder_pitch_joint' -> 'wam_shoulder_pitch_joint' (slashes are illegal in USD names)
    xml = re.sub(r'(?<![A-Za-z])(name|link)="([^"]*)"', lambda m: '%s="%s"' % (m.group(1), m.group(2).replace("/", "_")), xml)
    # collision meshes: use the fine STL (importer turns it into a convex hull)
    xml = re.sub(r'([A-Za-z_]+?)_convex(?:_decomposition)?\.dae', r'\1_fine.stl', xml)
    # the original efforts are tiny (cable-driven arm); give the sim drives headroom
    xml = re.sub(r'effort="[^"]*"', 'effort="200"', xml)
    return xml


def finger(side, sign):
    return f"""
  <link name="wam_finger_{side}_link">
    <inertial><origin xyz="0 0 0.05"/><mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>
    <visual><origin xyz="0 0 0.05"/><geometry><box size="0.012 0.012 0.10"/></geometry></visual>
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.012 0.012 0.10"/></geometry></collision>
  </link>
  <joint name="wam_finger_{side}_joint" type="prismatic">
    <parent link="wam_wrist_palm_stump_link"/><child link="wam_finger_{side}_link"/>
    <origin xyz="0 {sign*0.006} 0"/><axis xyz="0 {sign} 0"/>
    <limit lower="0" upper="{GRIP_OPEN}" effort="200" velocity="0.5"/>
  </joint>"""


GRIPPER = """
  <link name="wam_tool_link">
    <inertial><mass value="0.01"/><inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <joint name="wam_tool_joint" type="fixed">
    <parent link="wam_wrist_palm_stump_link"/><child link="wam_tool_link"/><origin xyz="0 0 0.10"/>
  </joint>""" + finger("l", +1) + finger("r", -1)


def write_urdf():
    xml = sanitize(expand_xacro())
    xml = xml.replace("</robot>", GRIPPER + "\n</robot>")
    pretty = minidom.parseString(xml).toprettyxml(indent="  ")
    assert "${" not in pretty, "unresolved xacro expression left in URDF"
    URDF_PATH.write_text(pretty)
    print("[wam] wrote", URDF_PATH)


def write_lula_yaml():
    lines = ["api_version: 1.0", "cspace:"]
    lines += ["    - " + j for j in ARM_JOINTS]
    lines += ["root_link: " + BASE_LINK,
              "default_q: [" + ", ".join(f"{q:.3f}" for q in HOME_Q) + "]",
              "acceleration_limits: [" + ", ".join(["10.0"] * 7) + "]",
              "jerk_limits: [" + ", ".join(["200.0"] * 7) + "]",
              "cspace_to_urdf_rules:"]
    for j in FINGER_JOINTS:
        lines.append(f"    - {{name: {j}, rule: fixed, value: 0.0}}")
    LULA_YAML.write_text("\n".join(lines) + "\n")
    print("[wam] wrote", LULA_YAML)


def import_to_usd():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.asset.importer.urdf")
    app.update()
    cfg_ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False      # keep wam_tool_link as a frame
    cfg.convex_decomp = False
    cfg.import_inertia_tensor = True
    cfg.fix_base = True
    cfg.self_collision = False
    cfg.make_default_prim = True
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    ok, prim = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(URDF_PATH), import_config=cfg, dest_path=str(USD_PATH))
    print("[wam] import ok=%s prim=%s -> %s" % (ok, prim, USD_PATH))
    app.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf-only", action="store_true")
    a = ap.parse_args()
    download_repo()
    write_urdf()
    write_lula_yaml()
    if not a.urdf_only:
        import_to_usd()
