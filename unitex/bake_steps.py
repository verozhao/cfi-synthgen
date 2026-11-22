"""
Shorter versions of UniTEX's bake step (CustomRGBTextureFullPipeline.step_2_ablition) for run_unitex_fast.py,
photo_front.py and back_panel.py, and the material of the baked model (glb_material, material).

step_2_ablition bakes twice: first without LTM (reprojection only) into cache/wo_LTM, then with LTM inpainting into
cache/w_LTM, and cache/textured_mesh.glb ends up as the second one. The first bake does not go into the result. It
costs 7 to 10 s of a 512 or 1024 px run and about 27 s of a photo_front.py bake at 4096 px. Only the walkthrough
page reads cache/wo_LTM (walkthrough_figures.py, walkthrough_page.py, runs/walkthrough_renders.sh show the stage
before LTM), so a run meant for that page keeps both bakes.
"""

import os
import shutil


def one_bake(pipe):
    """Only the bake with LTM. cache/wo_LTM is not written."""
    if pipe.super_resolutions:
        raise SystemExit("one_bake: not written for super_resolutions")

    def step(cache_dir, input_image_path, input_mesh_path, clear_cache=False, *args, **kwargs):
        j = os.path.join
        os.makedirs(j(cache_dir, "w_LTM"), exist_ok=True)
        pipe.sampling_on_mesh(cache_dir, j(cache_dir, "processed_mesh.obj"))
        pipe.infer_field(
            j(cache_dir, "w_LTM"), j(cache_dir, "processed_mesh.obj"), j(cache_dir, "sharp_pcd_fps.ply"),
            j(cache_dir, "coarse_pcd_fps.ply"), j(cache_dir, "mv_alpha.png"), j(cache_dir, "mv_ccm.png"),
            j(cache_dir, "mv_rgb.png"), base_or_inpainting=False, test_query_field=False, save_point_clouds=True)
        pipe.reproject_and_query_field(j(cache_dir, "w_LTM"), j(cache_dir, "processed_mesh.obj"),
                                       j(cache_dir, "mv_rgb.png"), j(cache_dir, "camera_info.pth"), inpainting=True)
        pipe.export_video(j(cache_dir, "w_LTM"), j(cache_dir, "w_LTM/textured_mesh.glb"), "textured_mesh.mp4")
        shutil.copy(j(cache_dir, "w_LTM/textured_mesh.glb"), j(cache_dir, "textured_mesh.glb"))
    pipe.step_2_ablition = step


def no_bake(pipe):
    """Views only (cache/mv_rgb.png), for a photo_front.py run next, which bakes itself.

    UniTEX's __call__ and run_unitex.py copy cache/textured_mesh.glb, so the untextured input mesh stands in for it
    and cache/NO_BAKE.txt says so.
    """
    def step(cache_dir, input_image_path, input_mesh_path, clear_cache=False, *args, **kwargs):
        shutil.copy(input_mesh_path, os.path.join(cache_dir, "textured_mesh.glb"))
        with open(os.path.join(cache_dir, "NO_BAKE.txt"), "w") as f:
            f.write("run_unitex_fast.py --no-bake: textured_mesh.glb is the untextured input mesh, "
                    "bake with photo_front.py\n")
    pipe.step_2_ablition = step


def fps_height(height=9, min_points=1_000_000):
    """Tree height of the farthest point sampling that picks the LTM's input points, for large point sets.

    UniTEX picks the LTM's 32 768 input points from all texels with fpsample.bucket_fps_kdline_sampling at tree
    height 5. At a 4096 px texture that is 11 million points and 43 s of a 58 s bake. fpsample recommends height 9
    for large inputs, which takes 5 s. UniTEX starts the sampling at a random point, so the sample differs from run
    to run either way. For the same start, height 7 picks the points of height 5 (13 s) and height 9 another
    farthest point sample. Point sets under min_points keep the height the caller asks for.
    """
    import fpsample
    orig = getattr(fpsample.bucket_fps_kdline_sampling, "_unwrapped", fpsample.bucket_fps_kdline_sampling)

    def sampling(pc, n_samples, h, start_idx=None):
        if len(pc) >= min_points:
            h = max(h, height)
        return orig(pc, n_samples, h, start_idx)
    sampling._unwrapped = orig
    fpsample.bucket_fps_kdline_sampling = sampling


def glb_material(path, roughness=None, drop_paths=True):
    """Rewrite the JSON part of a binary glTF in place. Returns what was changed, {} when nothing was.

    UniTEX's exporter writes every model fully matte (roughnessFactor 1.0, TextureTools structure.py) and keeps the
    absolute path of the mesh it read in the mesh's extras. roughness sets the roughnessFactor of every material
    without a metallicRoughnessTexture. drop_paths removes file_path from every extras. The binary part (geometry
    and texture) is kept byte for byte.
    """
    import json
    import struct
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 20:
        raise ValueError(f"{path} is not a binary glTF file")
    magic, version, total = struct.unpack("<4sII", raw[:12])
    n, kind = struct.unpack("<I4s", raw[12:20])
    if magic != b"glTF" or version != 2 or kind != b"JSON":
        raise ValueError(f"{path} is not a binary glTF 2 file")
    doc = json.loads(raw[20:20 + n].decode("utf-8"))
    rest = raw[20 + n:total]                      # the binary chunk with its header
    changed = {}
    if roughness is not None:
        for m in doc.get("materials", []):
            pbr = m.setdefault("pbrMetallicRoughness", {})
            if "metallicRoughnessTexture" not in pbr and pbr.get("roughnessFactor", 1.0) != float(roughness):
                pbr["roughnessFactor"] = float(roughness)
                changed["roughness"] = float(roughness)
    if drop_paths:
        def walk(node):
            if isinstance(node, dict):
                extras = node.get("extras")
                if isinstance(extras, dict) and "file_path" in extras:
                    del extras["file_path"]
                    changed["paths_dropped"] = changed.get("paths_dropped", 0) + 1
                    if not extras:
                        del node["extras"]
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(doc)
    if not changed:
        return changed
    body = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    body += b" " * (-len(body) % 4)
    out = (struct.pack("<4sII", b"glTF", 2, 20 + len(body) + len(rest)) + struct.pack("<I4s", len(body), b"JSON")
           + body + rest)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)
    return changed


def material(pipe, roughness, drop_paths=True):
    """The baked models get this roughness and lose the source path (glb_material), right after the bake step.

    UniTEX's __call__ copies cache/textured_mesh.glb to the run folder after the step, so that copy has them too.
    Call it after one_bake. Not for no_bake, whose placeholder is the input mesh.
    """
    step = pipe.step_2_ablition

    def baked(cache_dir, *args, **kwargs):
        out = step(cache_dir, *args, **kwargs)
        for rel in ("textured_mesh.glb", "w_LTM/textured_mesh.glb", "wo_LTM/textured_mesh.glb"):
            p = os.path.join(cache_dir, rel)
            if os.path.exists(p):
                glb_material(p, roughness, drop_paths)
        return out
    pipe.step_2_ablition = baked
