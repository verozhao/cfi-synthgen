"""
Angled views of finished GLBs for checking texture quality by eye: lit and albedo renders from four corners
(front-left, front-right, back-left, back-right, 20 deg up), perspective, 30 deg field of view. Uses mvgen.py's own
import, normalisation (no yaw), three-point lights and render passes, so the views match its straight views
(mvgen.py --mode views). Runs in the Blender Python (bpy), CPU by default. Skips models already rendered.

  <bpy python> unitex/quality_views.py <list: key path per line> <out dir> [res] [samples] [threads]
Writes <out dir>/<key>/view_{fl,fr,bl,br}.png (albedo) and view_*_lit.png.
"""
import math
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import bpy  # noqa: E402,F401
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import mvgen  # noqa: E402

CAMS = (("fl", 35.0, 20.0), ("fr", -35.0, 20.0), ("bl", 145.0, 20.0), ("br", -145.0, 20.0))
FOV = 30.0


def main(argv):
    lst, out_root = argv[0], pathlib.Path(argv[1])
    res = int(argv[2]) if len(argv) > 2 else 1024
    samples = int(argv[3]) if len(argv) > 3 else 32
    threads = int(argv[4]) if len(argv) > 4 else 12
    for line in open(lst):
        line = line.split("#")[0].strip()
        if not line:
            continue
        key, glb = line.split(None, 1)
        od = out_root / key
        if (od / "view_br_lit.png").exists():
            continue
        od.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="qv_")
        try:
            obj, _, textured, _ = mvgen.prepare_object(glb.strip(), 0.0, "cpu")
            scene = bpy.context.scene
            r = float(np.linalg.norm(mvgen.mesh_vertices(obj), axis=1).max())
            dist = r / math.sin(math.radians(FOV) / 2) * 1.05
            mvgen.setup_lighting(mvgen.object_rng(0, key), [])
            mvgen.set_shaded_settings(scene, res, samples, denoise=True)
            scene.render.threads_mode = "FIXED"
            scene.render.threads = threads
            names = mvgen.configure_passes(scene, ("image", "albedo"), tmp)
            for name, az, el in CAMS:
                mvgen.set_camera(scene.camera, mvgen.look_at(mvgen.direction(az, el) * dist), fov_deg=FOV)
                lit = mvgen.decode_lit(mvgen.render_passes(scene, names, tmp))
                Image.fromarray(lit["rgb"], "RGBA").save(od / f"view_{name}_lit.png")
                Image.fromarray(lit["albedo"], "RGBA").save(od / f"view_{name}.png")
            print(f"[{key}] done, textured {textured}", flush=True)
        except Exception as e:  # keep going with the next model
            print(f"[{key}] FAILED {e!r}", flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
    os._exit(0)  # skip the bpy teardown, which can dump core after all renders are saved
