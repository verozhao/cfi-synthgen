"""
Textured GLBs -> UniTEX-FLUX multi-view training data, plus eval renders of chosen views

Steps (--mode train):
  1. import each GLB welded and smooth, bake node transforms, optional yaw about the up axis,
     normalize to bbox centre at the origin and longest half-extent 0.95 (baked into the mesh)
  2. geometry pass per raw view (1 sample, no AA): CCM, camera-space normal, hard mask, UV
  3. lit pass per raw view (random three-point lights or a random HDRI): rgb + albedo AOV
  4. render_random reference candidates near the front, framed like UniTEX preprocess
  5. caption, metadata.json (written last, marks the object complete), training_uid.json
     rebuilt from disk so sharded runs merge

--mode views renders chosen raw views of one GLB (--glb) or many (--glb-list) with the same
cameras, normalization and yaw flags: unlit albedo RGBA, optionally lit and geometry maps.
--mode index only rebuilds training_uid.json.

Layout, frames and encodings: unitex/DESIGN.md and unitex/common.py.
Normalized Blender frame of every output: p = (Rz(yaw) @ G2B @ p_glb - centre) * factor.
"""

import argparse
import ctypes
import json
import math
import os
import pathlib
import random
import shutil
import sys
import tempfile
import time
import traceback
import zlib

import bpy
import numpy as np
from mathutils import Matrix
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitex import common as C


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_EXCLUDE = HERE / "unitex" / "bad_orientation_skus.txt"
CAPTION = "[MVFLUX]"          # UniTEX inference prompt

# CFI-3DGen's can unwrap (uv6_v2.py:364) centres the FRONT artwork 45 deg off +Z towards +X,
# so cans need -45 deg about the up axis to face raw view 0. Verified on Red Bull and Jif.
YAW_POLICIES = {
    "none":     {},
    "cfi3dgen": {"can": -45.0, "can_jar": -45.0},
}
DEFAULT_YAW_POLICY = {"train": "cfi3dgen", "views": "none"}

GREY_LINEAR = (0.5, 0.5, 0.5, 1.0)      # base colour for meshes without a material
PLACEHOLDER_MAPS = {                    # 1x1 maps, only needed by the UniTEX-FLUX assert
    "bump":      ("RGB", (128, 128, 255)),
    "metallic":  ("L", 0),
    "roughness": ("L", 153),
}

# Cycles area light facing a white Lambertian surface at distance d: pixel = P / (pi^2 d^2).
# Measured 0.0992 * P / d^2 with bpy 4.2 (probe_light.py), 1 / pi^2 = 0.1013.
AREA_LIGHT_K = 1.0 / math.pi ** 2

REF_ORTHO_FRACTION = 0.25     # share of render_random views that are orthographic front views
REF_FRAME_SCALE = 0.95        # UniTEX preprocess_reference_image(scale=0.95)

PASS_SOCKETS = {"image": "Image", "position": "Position", "normal": "Normal",
                "uv": "uv", "albedo": "albedo"}

GPU_BACKENDS = {
    "auto":  ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"),
    "cuda":  ("CUDA",),
    "optix": ("OPTIX",),
    "metal": ("METAL",),
}

VERBOSE = False


# ─────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────

class quiet_stdout:
    """Silence Blender's C-level stdout (render progress, importer chatter) unless --verbose."""

    def __enter__(self):
        self.fd = None
        if VERBOSE:
            return self
        sys.stdout.flush()
        _flush_c_stdout()
        self.fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            sys.stdout.flush()
            _flush_c_stdout()      # else buffered C output lands after the fd is restored
            os.dup2(self.fd, 1)
            os.close(self.fd)
        return False


def _flush_c_stdout():
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


def object_rng(seed, key):
    # Stable per object, so sharding or resuming never changes an object's lighting or views.
    return random.Random(zlib.crc32(f"{seed}:{key}".encode()))


def load_manifest_entry(glb_path):
    sidecar = pathlib.Path(glb_path).parent / "manifest_entry.json"
    if not sidecar.exists():
        return None
    with open(sidecar) as f:
        return json.load(f)


def write_json_atomic(path, data):
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):
    """Blender cam2world (looks down local -Z, image up = local +Y) with no roll."""
    eye = np.asarray(eye, dtype=np.float64)
    z = eye - np.asarray(target, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross((0.0, 1.0, 0.0), z)
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, eye
    return m


def direction(azimuth_deg, elevation_deg):
    """Unit vector from the object to a viewer. Azimuth 0 = front (-Y_b), +90 = right (+X)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])


def kelvin_to_rgb(kelvin):
    """Approximate blackbody colour (Tanner Helland fit), linear, max channel 1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    srgb = [min(max(v, 0.0), 255.0) / 255.0 for v in (r, g, b)]
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in srgb]
    m = max(lin)
    return tuple(v / m for v in lin)


# ─────────────────────────────────────────────────────────────────────────
# Output encoding (numpy, rows top-down)
# ─────────────────────────────────────────────────────────────────────────

def srgb_oetf(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def quantize(x):
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def quantize_data(x):
    # CCM and normals truncate like UniTEX export_condition ((x * 255).astype(uint8)),
    # so training conditions carry the same quantization as inference conditions.
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def unpremultiply(rgb, alpha):
    a = alpha[..., None]
    return np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0)


def decode_geometry(passes, c2w):
    """1-sample passes -> hard mask, Blender-frame position, camera-frame normal, UV (Blender, v up)."""
    mask = passes["image"][..., 3] > 0.5
    m3 = mask[..., None]
    pos = np.where(m3, passes["position"][..., :3], 0.0)
    n_cam = passes["normal"][..., :3] @ np.asarray(c2w, dtype=np.float64)[:3, :3]   # R^T n per pixel
    n_cam /= np.maximum(np.linalg.norm(n_cam, axis=-1, keepdims=True), 1e-8)
    n_cam = np.where(m3, n_cam, 0.0)
    uv = np.where(m3, passes["uv"][..., :2], 0.0) if "uv" in passes else None
    return {"mask": mask, "pos": pos, "normal_cam": n_cam, "uv": uv}


def encode_ccm_png(geo):
    rgb = quantize_data((geo["pos"] + 1.0) / 2.0)
    rgb[~geo["mask"]] = 0
    return np.dstack([rgb, geo["mask"].astype(np.uint8) * 255])


def encode_normal_png(geo):
    rgb = quantize_data((geo["normal_cam"] + 1.0) / 2.0)
    rgb[~geo["mask"]] = 255
    return rgb


def decode_lit(passes):
    """Premultiplied linear EXR passes -> straight-alpha sRGB RGBA for rgb and albedo (uint8)."""
    img = passes["image"]
    alpha = np.clip(img[..., 3], 0.0, 1.0)
    a8 = quantize(alpha)
    out = {}
    for key, name in (("rgb", "image"), ("albedo", "albedo")):
        if name not in passes:
            continue
        rgb = quantize(srgb_oetf(unpremultiply(passes[name][..., :3], alpha)))
        rgb[a8 == 0] = 0
        out[key] = np.dstack([rgb, a8])
    return out


def on_white(rgba):
    # Straight colour wherever alpha > 0 and white elsewhere: the loader's own
    # composite with the *_rgb alpha (rgb * a + (1 - a)) then happens exactly once.
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 255
    return rgb


def frame_like_unitex(images_rgba, res, scale=REF_FRAME_SCALE):
    """UniTEX preprocess(): tight alpha bbox, longer side scale * res, centred, transparent canvas.

    Every image in `images_rgba` gets the crop of the first one's alpha. Returns the framed
    images and the parameters that map render pixels to reference pixels.
    """
    alpha = images_rgba[0][..., 3]
    rows, cols = np.nonzero(alpha.sum(1))[0], np.nonzero(alpha.sum(0))[0]
    if len(rows) == 0:
        raise RuntimeError("empty alpha in a reference view")
    x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())
    dy, dx = max(y2 - y1, 1), max(x2 - x1, 1)
    s = min(res * scale / dy, res * scale / dx)
    ht, wt = int(dy * s), int(dx * s)
    ox, oy = int((res - wt) / 2), int((res - ht) / 2)
    framed = []
    for rgba in images_rgba:
        # PIL resizes RGBA premultiplied internally, so no dark fringes at the silhouette.
        crop = Image.fromarray(rgba, "RGBA").crop((x1, y1, x2, y2)).resize((wt, ht), Image.BICUBIC)
        canvas = Image.new("RGBA", (res, res), (0, 0, 0, 0))
        canvas.paste(crop, (ox, oy))
        framed.append(np.asarray(canvas))
    params = {"bbox": [x1, y1, x2, y2], "scale": s, "size": [wt, ht], "offset": [ox, oy],
              "render_res": int(alpha.shape[0]), "res": res}
    return framed, params


# ─────────────────────────────────────────────────────────────────────────
# Scene reset + device
# ─────────────────────────────────────────────────────────────────────────

def _usable_device(dev, backend):
    # Cycles lists Intel iGPUs under Metal but they time out (Iris Plus, bpy 4.2).
    return dev.type == backend and not (backend == "METAL" and "intel" in dev.name.lower())


def apply_device(scene, device):
    """Select the Cycles device. read_factory_settings resets the prefs, so call after every reset."""
    scene.cycles.device = "CPU"
    if device == "cpu":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        prefs = None
    for backend in (GPU_BACKENDS[device] if prefs is not None else ()):
        try:
            prefs.compute_device_type = backend
        except TypeError:      # backend not compiled in / not available on this OS
            continue
        try:
            prefs.refresh_devices()
        except AttributeError:
            prefs.get_devices()
        gpus = [d for d in prefs.devices if _usable_device(d, backend)]
        if not gpus:
            continue
        ids = {d.id for d in gpus}
        for d in prefs.devices:
            d.use = d.id in ids
        scene.cycles.device = "GPU"
        return f"{backend} ({', '.join(sorted({d.name for d in gpus}))})"
    if device != "auto":
        raise RuntimeError(f"--device {device}: no usable {device.upper()} device")
    return "CPU"


def reset_scene(device):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    device_label = apply_device(scene, device)
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.frame_current = 1

    view_layer = scene.view_layers[0]
    view_layer.use_pass_position = True
    view_layer.use_pass_normal = True
    for name in ("uv", "albedo"):
        aov = view_layer.aovs.add()
        aov.name, aov.type = name, "COLOR"

    cam = bpy.data.objects.new("MVCam", bpy.data.cameras.new("MVCam"))
    cam.data.clip_start, cam.data.clip_end = 0.01, 100.0
    cam.data.sensor_fit = "AUTO"
    scene.collection.objects.link(cam)
    scene.camera = cam

    world = bpy.data.worlds.new("MVWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world
    return device_label


# ─────────────────────────────────────────────────────────────────────────
# GLB import + normalization + materials
# ─────────────────────────────────────────────────────────────────────────

def import_mesh(glb_path):
    """Import, bake every node transform into the mesh data, join into one object.

    The importer bakes Y-up -> Z-up into the vertices and sets rotation_mode QUATERNION,
    so transforms are applied to mesh data and matrix_world is left at identity.
    """
    if not os.path.isfile(glb_path):
        raise FileNotFoundError(glb_path)
    with quiet_stdout():
        bpy.ops.import_scene.gltf(filepath=str(glb_path), merge_vertices=True, import_shading="SMOOTH")
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    meshes = [o for o in scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {glb_path}")
    world_mats = [o.matrix_world.copy() for o in meshes]
    for obj, mw in zip(meshes, world_mats):
        if obj.data.users > 1:
            obj.data = obj.data.copy()
        obj.data.transform(mw)
        if mw.determinant() < 0:   # mirrored node: transform() flips winding
            for poly in obj.data.polygons:
                poly.flip()
        # import_shading SMOOTH still marks primitives without a NORMAL attribute flat
        # (sharp_face), which is every CFI GLB. TextureTools renders interpolated vertex
        # normals of the welded mesh, so drop it (custom normals from a NORMAL attribute stay).
        sharp = obj.data.attributes.get("sharp_face")
        if sharp is not None:
            obj.data.attributes.remove(sharp)
        obj.parent = None
        obj.matrix_basis = Matrix.Identity(4)
    for obj in list(scene.objects):
        if obj.type not in ("MESH", "CAMERA") or (obj.type == "CAMERA" and obj.name != "MVCam"):
            bpy.data.objects.remove(obj, do_unlink=True)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active if len(meshes) > 1 else meshes[0]
    obj.name = "MVObject"
    return obj


def mesh_vertices(obj):
    mesh = obj.data
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64)


def normalize_object(obj, yaw_deg):
    """Yaw about Blender Z (= glTF up), then bbox centre -> origin and longest half-extent -> 0.95."""
    if yaw_deg:
        obj.data.transform(Matrix.Rotation(math.radians(yaw_deg), 4, "Z"))
    centre, factor = C.normalize_params(mesh_vertices(obj))
    obj.data.transform(Matrix.Scale(factor, 4) @ Matrix.Translation(-centre))
    obj.data.update()
    return {"centre": [float(v) for v in centre], "factor": float(factor), "frame": "blender_after_yaw",
            "formula": "p = (Rz(yaw_deg) @ G2B @ p_glb - centre) * factor"}


def _set_aov_name(node, name):
    if hasattr(node, "aov_name"):
        node.aov_name = name
    else:
        node.name = name


def prepare_materials(obj):
    """Metallic 0 and opaque (synthgen flatten_metals), plus an 'albedo' AOV fed by the base colour.

    The AOV takes whatever feeds Principled Base Color (the glTF Image Texture Color output,
    or a factor multiply in front of it), so it is the linear texture colour. Returns True
    when the object has a texture-driven base colour.
    """
    if not any(slot.material for slot in obj.material_slots):
        mat = bpy.data.materials.new("MVGrey")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = GREY_LINEAR
        bsdf.inputs["Roughness"].default_value = 0.6
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    textured = False
    for mat in {slot.material for slot in obj.material_slots if slot.material}:
        mat.use_nodes = True
        nt = mat.node_tree
        src = None
        for bsdf in [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"]:
            for name, value in (("Metallic", 0.0), ("Alpha", 1.0)):
                sock = bsdf.inputs.get(name)
                if sock is None:
                    continue
                for link in list(sock.links):
                    nt.links.remove(link)
                sock.default_value = value
            base = bsdf.inputs["Base Color"]
            if src is None:
                src = base.links[0].from_socket if base.is_linked else tuple(base.default_value)
        if src is None:    # no Principled BSDF (e.g. unlit extension): first image texture
            tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
            src = tex.outputs["Color"] if tex is not None else GREY_LINEAR
        aov = nt.nodes.new("ShaderNodeOutputAOV")
        _set_aov_name(aov, "albedo")
        if isinstance(src, tuple):
            aov.inputs["Color"].default_value = src
        else:
            nt.links.new(src, aov.inputs["Color"])
            textured = True
    return textured


def geometry_material():
    """Override for the geometry pass: plain diffuse + a 'uv' AOV.

    It must be a BSDF: Cycles writes the Normal pass from BSDF closures, and with an
    emission-only surface both the Normal pass and the UV AOV come out wrong (measured).
    """
    mat = bpy.data.materials.new("MVGeometry")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    uv = nt.nodes.new("ShaderNodeUVMap")        # empty uv_map = the active render UV layer
    aov = nt.nodes.new("ShaderNodeOutputAOV")
    _set_aov_name(aov, "uv")
    nt.links.new(uv.outputs["UV"], aov.inputs["Color"])
    return mat


def prepare_object(glb_path, yaw_deg, device):
    device_label = reset_scene(device)
    obj = import_mesh(glb_path)
    norm = normalize_object(obj, yaw_deg)
    textured = prepare_materials(obj)
    return obj, norm, textured, device_label


# ─────────────────────────────────────────────────────────────────────────
# Lighting
# ─────────────────────────────────────────────────────────────────────────

def add_area_light(name, azimuth_deg, elevation_deg, distance, size, level, kelvin):
    light = bpy.data.lights.new(name, "AREA")
    light.size = size
    light.energy = level * distance ** 2 / AREA_LIGHT_K
    light.color = kelvin_to_rgb(kelvin)
    obj = bpy.data.objects.new(name, light)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = Matrix(look_at(direction(azimuth_deg, elevation_deg) * distance).tolist())
    return {"name": name, "azimuth_deg": azimuth_deg, "elevation_deg": elevation_deg, "distance": distance,
            "size": size, "level": level, "power_w": light.energy, "kelvin": kelvin}


def setup_lighting(rng, hdri_files):
    """Per-object lighting so lit != albedo (UniTEX's delight target needs real shading).

    With HDRIs: one random map, random rotation and strength. Otherwise a randomized
    three-point rig (key / fill / rim area lights) plus a dim neutral world, like the
    UniTEX author's delight renders. `level` is the linear value a white surface facing
    that light would reach.
    """
    world = bpy.context.scene.world
    nt = world.node_tree
    bg = nt.nodes["Background"]
    if hdri_files:
        path = rng.choice(hdri_files)
        rot, strength = rng.uniform(0.0, 360.0), rng.uniform(0.6, 1.5)
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(path), check_existing=True)
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(rot))
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = strength
        return {"type": "hdri", "path": str(path), "rotation_deg": rot, "strength": strength}

    side = rng.choice((-1.0, 1.0))
    key = rng.uniform(0.8, 1.2)
    lights = [
        add_area_light("key", side * rng.uniform(20, 65), rng.uniform(15, 50), rng.uniform(3.5, 5.0),
                       rng.uniform(1.0, 3.0), key, rng.uniform(4200, 7000)),
        add_area_light("fill", -side * rng.uniform(30, 80), rng.uniform(-5, 30), rng.uniform(3.5, 5.0),
                       rng.uniform(2.0, 4.0), key * rng.uniform(0.2, 0.5), rng.uniform(5000, 8000)),
        add_area_light("rim", rng.uniform(140, 220), rng.uniform(20, 60), rng.uniform(3.5, 5.0),
                       rng.uniform(1.0, 3.0), key * rng.uniform(0.3, 0.9), rng.uniform(5000, 9000)),
    ]
    # Ambient keeps shadowed faces (bottom, back) readable instead of near black.
    grey, strength = rng.uniform(0.85, 1.0), rng.uniform(0.06, 0.18)
    bg.inputs["Color"].default_value = (grey, grey, grey, 1.0)
    bg.inputs["Strength"].default_value = strength
    return {"type": "three_point", "lights": lights, "world_strength": strength}


# ─────────────────────────────────────────────────────────────────────────
# Render passes (single-layer EXR File Output, read back with bpy)
# ─────────────────────────────────────────────────────────────────────────

def configure_passes(scene, names, tmp_dir):
    # One float EXR per pass: the bpy wheel has no OpenImageIO, so multilayer EXR
    # cannot be read back, while bpy.data.images.load reads single-layer EXR.
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    out = tree.nodes.new("CompositorNodeOutputFile")
    out.base_path = str(tmp_dir)
    out.format.file_format = "OPEN_EXR"
    out.format.color_depth = "32"
    out.format.exr_codec = "NONE"
    out.format.color_mode = "RGBA"
    out.file_slots.clear()
    for name in names:
        out.file_slots.new(f"{name}_")
        tree.links.new(rl.outputs[PASS_SOCKETS[name]], out.inputs[len(out.inputs) - 1])
    return tuple(names)


def read_exr(path):
    img = bpy.data.images.load(str(path), check_existing=False)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1].astype(np.float64)    # Blender rows are bottom-up


def render_passes(scene, names, tmp_dir):
    with quiet_stdout():
        bpy.ops.render.render(write_still=False)
    out = {}
    for name in names:
        path = pathlib.Path(tmp_dir) / f"{name}_{scene.frame_current:04d}.exr"
        out[name] = read_exr(path)
        path.unlink()
    return out


def set_resolution(scene, res):
    scene.render.resolution_x = scene.render.resolution_y = int(res)


def set_geometry_settings(scene, res, override):
    # 1 sample at the pixel centre and no pixel filter, like nvdiffrast enable_antialis=False.
    c = scene.cycles
    c.samples = 1
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.pixel_filter_type = "BOX"
    c.filter_width = 0.01
    c.max_bounces = c.diffuse_bounces = c.glossy_bounces = 0
    c.transmission_bounces = c.volume_bounces = c.transparent_max_bounces = 0
    scene.view_layers[0].material_override = override
    set_resolution(scene, res)


def set_shaded_settings(scene, res, samples, denoise, lit=True):
    c = scene.cycles
    c.samples = int(samples)
    # Adaptive sampling watches the Combined pass only and would stop early on flat
    # shading while the albedo AOV still aliases fine label text.
    c.use_adaptive_sampling = False
    c.use_denoising = False
    if denoise:
        try:
            c.denoiser = "OPENIMAGEDENOISE"
            c.use_denoising = True
            if hasattr(c, "denoising_use_gpu"):
                c.denoising_use_gpu = c.device == "GPU"
        except TypeError:
            c.use_denoising = False
    c.pixel_filter_type = "BLACKMAN_HARRIS"
    c.filter_width = 1.5
    if lit:
        c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 6, 3, 3
        c.transmission_bounces, c.volume_bounces, c.transparent_max_bounces = 2, 0, 4
    else:
        c.max_bounces = c.diffuse_bounces = c.glossy_bounces = 0
        c.transmission_bounces = c.volume_bounces = c.transparent_max_bounces = 0
    scene.view_layers[0].material_override = None
    set_resolution(scene, res)


def set_camera(cam, c2w, ortho_scale=None, fov_deg=None):
    cam.matrix_world = Matrix(np.asarray(c2w, dtype=np.float64).tolist())
    cam.data.shift_x = cam.data.shift_y = 0.0
    if ortho_scale is not None:
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = float(ortho_scale)
    else:
        cam.data.type = "PERSP"
        cam.data.lens_unit = "FOV"
        cam.data.angle = math.radians(fov_deg)


def render_geometry_views(scene, views, res, tmp_dir):
    set_geometry_settings(scene, res, geometry_material())
    names = configure_passes(scene, ("image", "position", "normal", "uv"), tmp_dir)
    out = {}
    for i in views:
        set_camera(scene.camera, C.RAW_C2W[i], ortho_scale=C.ORTHO_SCALE)
        out[i] = decode_geometry(render_passes(scene, names, tmp_dir), C.RAW_C2W[i])
    return out


# ─────────────────────────────────────────────────────────────────────────
# Reference candidates (render_random)
# ─────────────────────────────────────────────────────────────────────────

def sample_reference_cameras(rng, n, verts):
    """Front-ish cameras looking at the origin: a few orthographic, the rest perspective.

    View 0 is the exact front orthographic view. UniTEX-FLUX weights candidates by the cosine
    between each camera z axis and raw view 0's, so the c2w written is the true camera pose.
    """
    r_sphere = float(np.linalg.norm(verts, axis=1).max())
    n_ortho = max(1, int(round(n * REF_ORTHO_FRACTION))) if n > 0 else 0
    cams = []
    for k in range(n):
        if k == 0:
            az, el = 0.0, 0.0
        elif k < n_ortho:
            az, el = rng.uniform(-15, 15), rng.uniform(-5, 15)
        else:
            az, el = rng.uniform(-60, 60), rng.uniform(-10, 30)
        d = direction(az, el)
        if k < n_ortho:
            c2w = look_at(d * C.RADIUS)
            q = verts @ c2w[:3, :3]
            half = float(np.abs(q[:, :2]).max())
            cams.append({"type": "ORTHO", "c2w": c2w, "ortho_scale": 2.0 * half * 1.04, "fov_deg": None,
                         "azimuth_deg": az, "elevation_deg": el, "distance": C.RADIUS})
        else:
            fov = rng.uniform(30, 50)
            dist = r_sphere / math.sin(math.radians(fov) / 2) * 1.02
            cams.append({"type": "PERSP", "c2w": look_at(d * dist), "ortho_scale": None, "fov_deg": fov,
                         "azimuth_deg": az, "elevation_deg": el, "distance": dist})
    return cams


# ─────────────────────────────────────────────────────────────────────────
# Mode train: one object -> render/, render_random/, caption/
# ─────────────────────────────────────────────────────────────────────────

OUTPUT_PATTERNS = ("[0-9][0-9][0-9][0-9]_*.png", "[0-9][0-9][0-9][0-9]_uv.npz", "metadata.json",
                   "data.mdb", "lock.mdb")


def clear_outputs(d):
    # Only mvgen's own files: text.json and other annotations written later are kept.
    for pattern in OUTPUT_PATTERNS:
        for p in d.glob(pattern):
            p.unlink()


def resolve_yaw(args, shape):
    if args.yaw_deg is not None:
        return float(args.yaw_deg)
    return float(YAW_POLICIES[args.yaw_policy].get(shape or "", 0.0))


def render_training_object(job, args, out_root, hdri_files):
    sku = job["sku"]
    rng = object_rng(args.seed, sku)
    yaw = resolve_yaw(args, job["shape"])
    t0 = time.time()
    obj, norm, textured, device_label = prepare_object(job["glb"], yaw, args.device)
    scene = bpy.context.scene
    lighting = setup_lighting(rng, hdri_files)
    if not textured:
        print(f"  [{sku}] warning: no texture feeds Base Color, albedo is a constant colour")

    render_dir = out_root / "render" / args.volume / sku
    random_dir = out_root / "render_random" / args.volume / sku
    caption_dir = out_root / "caption" / args.volume / sku
    for d in (render_dir, random_dir, caption_dir):
        d.mkdir(parents=True, exist_ok=True)
        clear_outputs(d)
    timings = {"setup": time.time() - t0}
    tmp_dir = tempfile.mkdtemp(prefix="mvgen_")
    try:
        t = time.time()
        geo = render_geometry_views(scene, range(6), args.res, tmp_dir)
        for i, g in geo.items():
            stem = render_dir / f"{i:04d}"
            Image.fromarray(encode_ccm_png(g), "RGBA").save(f"{stem}_nocs.png")
            Image.fromarray(encode_normal_png(g), "RGB").save(f"{stem}_normal.png")
            np.savez_compressed(f"{stem}_uv.npz", uv=g["uv"].astype(np.float16), mask=g["mask"])
        timings["geometry"] = time.time() - t

        t = time.time()
        set_shaded_settings(scene, args.res, args.samples, denoise=not args.no_denoise)
        names = configure_passes(scene, ("image", "albedo"), tmp_dir)
        for i in range(6):
            set_camera(scene.camera, C.RAW_C2W[i], ortho_scale=C.ORTHO_SCALE)
            lit = decode_lit(render_passes(scene, names, tmp_dir))
            stem = render_dir / f"{i:04d}"
            Image.fromarray(lit["rgb"], "RGBA").save(f"{stem}_rgb.png")
            Image.fromarray(on_white(lit["albedo"]), "RGB").save(f"{stem}_albedo.png")
            for kind, (mode, value) in PLACEHOLDER_MAPS.items():
                Image.new(mode, (1, 1), value).save(f"{stem}_{kind}.png")
        timings["lit"] = time.time() - t

        t = time.time()
        cams = sample_reference_cameras(rng, args.random_views, mesh_vertices(obj))
        render_res = int(round(args.ref_res * args.ref_supersample))
        set_shaded_settings(scene, render_res, args.samples, denoise=not args.no_denoise)
        framing = []
        for k, cam in enumerate(cams):
            set_camera(scene.camera, cam["c2w"], ortho_scale=cam["ortho_scale"], fov_deg=cam["fov_deg"])
            lit = decode_lit(render_passes(scene, names, tmp_dir))
            (rgb, albedo), params = frame_like_unitex([lit["rgb"], lit["albedo"]], args.ref_res)
            Image.fromarray(rgb, "RGBA").save(random_dir / f"{k:04d}_rgb.png")
            Image.fromarray(on_white(albedo), "RGB").save(random_dir / f"{k:04d}_albedo.png")
            framing.append(params)
        write_json_atomic(random_dir / "metadata.json", {
            "cam2world_matrixs": [c["c2w"].tolist() for c in cams],
            "camera_type": [c["type"] for c in cams],
            "fov_deg": [c["fov_deg"] for c in cams],
            "ortho_scale": [c["ortho_scale"] for c in cams],
            "azimuth_deg": [c["azimuth_deg"] for c in cams],
            "elevation_deg": [c["elevation_deg"] for c in cams],
            "distance": [c["distance"] for c in cams],
            "framing": framing,
            "framing_note": "ref_xy = offset + (render_xy - bbox[:2]) * size / (bbox[2:] - bbox[:2]); "
                            "cameras are Blender c2w in the normalized frame, all looking at the origin",
            "sku": sku,
        })
        timings["random"] = time.time() - t

        (caption_dir / "prompt.txt").write_text(CAPTION)
        timings["total"] = time.time() - t0
        # Written last: its presence marks the object complete (--skip-existing, index).
        write_json_atomic(render_dir / "metadata.json", {
            "cam2world_matrixs": C.RAW_C2W.tolist(),
            "view_names": list(C.RAW_VIEW_NAMES),
            "full_index": list(C.FULL_INDEX),
            "res": args.res,
            "ortho_scale": C.ORTHO_SCALE,
            "normalize": norm,
            "yaw_deg": yaw,
            "sku": sku,
            "shape": job["shape"],
            "title": job["title"],
            "source_glb": str(job["glb"]),
            "lighting": lighting,
            "samples": args.samples,
            "seed": args.seed,
            "encoding": {
                "nocs": "RGB = trunc(255 * (p_blender + 1) / 2), A = hard mask",
                "normal": "RGB = trunc(255 * (n_cam + 1) / 2), Blender camera frame, background white",
                "rgb": "straight-alpha sRGB (Standard view transform)"
                       + ("" if args.no_denoise else ", OIDN denoised"),
                "albedo": "sRGB of the linear base-colour AOV, straight where alpha > 0, white elsewhere",
                "uv": "float16 Blender UV (u right, v up): texture pixel = (u * W, (1 - v) * H)",
            },
            "textured": textured,
            "device": device_label,
            "timings_s": timings,
        })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return timings


def discover_jobs(glbs_dir, glb_name):
    jobs, seen = [], set()
    for glb in sorted(pathlib.Path(glbs_dir).rglob(glb_name)):
        entry = load_manifest_entry(glb) or {}
        sku = str(entry.get("sku") or glb.parent.name).replace("/", "_")
        if sku in seen:
            print(f"  [{sku}] duplicate sku, keeping the first GLB, skipping {glb}")
            continue
        seen.add(sku)
        jobs.append({"sku": sku, "glb": glb, "shape": entry.get("shape"), "title": entry.get("title")})
    return jobs


def read_exclude(path):
    if not path or str(path).lower() == "none" or not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}


def parse_shard(text):
    i, n = (int(v) for v in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= i < N")
    return i, n


def rebuild_training_uids(out_root):
    """training_uid.json from every uid whose render/ and render_random/ metadata exist."""
    uids = []
    render_root = out_root / "render"
    for meta in sorted(render_root.glob("*/*/metadata.json")):
        uid = meta.parent.relative_to(render_root).as_posix()
        if (out_root / "render_random" / uid / "metadata.json").exists():
            uids.append(uid)
    out_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_root / "training_uid.json", uids)
    return uids


def log_error(out_root, key, exc_text):
    with open(out_root / "mvgen_errors.log", "a") as f:
        f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {key}\n{exc_text}\n")


def run_train(args, hdri_files):
    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs(args.glbs, args.glb_name)
    exclude = read_exclude(args.exclude)
    if args.skus:
        wanted = {s.strip() for s in args.skus.split(",") if s.strip()}
        jobs = [j for j in jobs if j["sku"] in wanted]
    n_excluded = sum(j["sku"] in exclude for j in jobs)
    jobs = [j for j in jobs if j["sku"] not in exclude]
    if args.shard:
        i, n = args.shard
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    if args.skip_existing:
        jobs = [j for j in jobs if not (out_root / "render" / args.volume / j["sku"] / "metadata.json").exists()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"mvgen train: {len(jobs)} objects to render ({n_excluded} excluded), yaw policy {args.yaw_policy}, "
          f"res {args.res}, samples {args.samples}, random views {args.random_views}")
    n_ok, t_run = 0, time.time()
    for k, job in enumerate(jobs):
        print(f"  [{job['sku']}] {k + 1}/{len(jobs)} shape={job['shape']} yaw={resolve_yaw(args, job['shape']):g}")
        try:
            tm = render_training_object(job, args, out_root, hdri_files)
            n_ok += 1
            print(f"  [{job['sku']}] done in {tm['total']:.1f} s (geometry {tm['geometry']:.1f}, "
                  f"lit {tm['lit']:.1f}, random {tm['random']:.1f})")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [{job['sku']}] FAILED, traceback in mvgen_errors.log\n{tb}")
            log_error(out_root, job["sku"], tb)
    uids = rebuild_training_uids(out_root)
    print(f"mvgen train: {n_ok}/{len(jobs)} rendered in {time.time() - t_run:.1f} s, "
          f"training_uid.json lists {len(uids)} uids")
    return len(jobs) - n_ok


# ─────────────────────────────────────────────────────────────────────────
# Mode views: raw views of one GLB for evaluation
# ─────────────────────────────────────────────────────────────────────────

def render_eval_object(key, glb, out_dir, args, hdri_files):
    entry = load_manifest_entry(glb) or {}
    shape = entry.get("shape")
    yaw = resolve_yaw(args, shape)
    t0 = time.time()
    obj, norm, textured, device_label = prepare_object(glb, yaw, args.device)
    scene = bpy.context.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    views = args.views
    tmp_dir = tempfile.mkdtemp(prefix="mvgen_")
    lighting = None
    try:
        if args.with_geometry:
            for i, g in render_geometry_views(scene, views, args.res, tmp_dir).items():
                Image.fromarray(encode_ccm_png(g), "RGBA").save(out_dir / f"view_{i:02d}_nocs.png")
                Image.fromarray(encode_normal_png(g), "RGB").save(out_dir / f"view_{i:02d}_normal.png")
                Image.fromarray(g["mask"].astype(np.uint8) * 255, "L").save(out_dir / f"view_{i:02d}_alpha.png")
        if args.lit:
            lighting = setup_lighting(object_rng(args.seed, key), hdri_files)
            set_shaded_settings(scene, args.res, args.samples, denoise=not args.no_denoise)
        else:
            set_shaded_settings(scene, args.res, args.albedo_samples, denoise=False, lit=False)
        names = configure_passes(scene, ("image", "albedo"), tmp_dir)
        for i in views:
            set_camera(scene.camera, C.RAW_C2W[i], ortho_scale=C.ORTHO_SCALE)
            lit = decode_lit(render_passes(scene, names, tmp_dir))
            Image.fromarray(lit["albedo"], "RGBA").save(out_dir / f"view_{i:02d}.png")
            if args.lit:
                Image.fromarray(lit["rgb"], "RGBA").save(out_dir / f"view_{i:02d}_lit.png")
        write_json_atomic(out_dir / "views.json", {
            "id": key, "source_glb": str(glb), "shape": shape, "views": list(views), "res": args.res,
            "cam2world_matrixs": {str(i): C.RAW_C2W[i].tolist() for i in views},
            "ortho_scale": C.ORTHO_SCALE, "normalize": norm, "yaw_deg": yaw, "textured": textured,
            "albedo_samples": None if args.lit else args.albedo_samples,
            "lit_samples": args.samples if args.lit else None, "lighting": lighting,
            "with_geometry": bool(args.with_geometry), "device": device_label,
            "time_s": time.time() - t0,
        })
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return time.time() - t0


def run_views(args, hdri_files):
    out_root = pathlib.Path(args.out)
    if args.glb:
        jobs = [(pathlib.Path(args.glb).parent.name, pathlib.Path(args.glb), out_root)]
    else:
        jobs = []
        with open(args.glb_list) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    key, path = line.split(None, 1)
                    jobs.append((key, pathlib.Path(path.strip()), out_root / key))
    exclude = read_exclude(args.exclude) if args.exclude else set()
    jobs = [j for j in jobs if j[0] not in exclude]
    if args.shard:
        i, n = args.shard
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    if args.skip_existing:
        jobs = [j for j in jobs if not (j[2] / "views.json").exists()]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"mvgen views: {len(jobs)} GLBs, views {list(args.views)}, res {args.res}, "
          f"yaw policy {args.yaw_policy}, lit {bool(args.lit)}")
    n_failed = 0
    for key, glb, out_dir in jobs:
        try:
            dt = render_eval_object(key, glb, out_dir, args, hdri_files)
            print(f"  [{key}] done in {dt:.1f} s -> {out_dir}")
        except Exception:
            n_failed += 1
            tb = traceback.format_exc()
            print(f"  [{key}] FAILED\n{tb}")
            out_root.mkdir(parents=True, exist_ok=True)
            log_error(out_root, key, tb)
    return n_failed


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--mode", choices=("train", "views", "index"), default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--glbs", help="train: folder searched recursively for --glb-name")
    parser.add_argument("--glb-name", default="textured.glb")
    parser.add_argument("--glb", help="views: one GLB")
    parser.add_argument("--glb-list", help="views: text file with 'id path' per line")
    parser.add_argument("--volume", default="cfi", help="uid prefix: uid = <volume>/<sku>")
    parser.add_argument("--skus", help="comma-separated sku filter")
    parser.add_argument("--exclude", default=None,
                        help=f"sku list to skip (train default {DEFAULT_EXCLUDE.relative_to(HERE)}, 'none' disables)")
    parser.add_argument("--yaw-policy", choices=tuple(YAW_POLICIES), default=None,
                        help="default cfi3dgen for train, none for views")
    parser.add_argument("--yaw-deg", type=float, default=None, help="override the policy for every object")
    parser.add_argument("--res", type=int, default=C.VIEW_RES)
    parser.add_argument("--samples", type=int, default=64, help="lit pass samples")
    parser.add_argument("--albedo-samples", type=int, default=32, help="views mode without --lit")
    parser.add_argument("--random-views", type=int, default=20)
    parser.add_argument("--ref-res", type=int, default=512)
    parser.add_argument("--ref-supersample", type=float, default=1.25,
                        help="render_random views are rendered at ref-res * this, then framed")
    parser.add_argument("--no-denoise", action="store_true",
                        help="skip OIDN (about 2.4 s per 512 px view on a laptop CPU)")
    parser.add_argument("--hdri-dir", help="random .exr/.hdr per object instead of three-point lights")
    parser.add_argument("--views", default="0,1,2,3,4,5", help="views mode: raw view indices")
    parser.add_argument("--lit", action="store_true", help="views mode: also write view_XX_lit.png")
    parser.add_argument("--with-geometry", action="store_true", help="views mode: nocs / normal / alpha")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--shard", type=parse_shard, default=None, help="i/N: every N-th object from i")
    parser.add_argument("--device", choices=tuple(["cpu"] + list(GPU_BACKENDS)), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="keep Blender's render / import logs")
    args = parser.parse_args(argv)

    global VERBOSE
    VERBOSE = args.verbose
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.yaw_policy is None:
        args.yaw_policy = DEFAULT_YAW_POLICY.get(args.mode, "none")
    args.views = tuple(int(v) for v in args.views.split(",") if v.strip())
    if any(not 0 <= v < 6 for v in args.views):
        parser.error("--views must be raw indices 0..5")
    hdri_files = []
    if args.hdri_dir:
        hdri_files = sorted(str(p) for p in pathlib.Path(args.hdri_dir).rglob("*")
                            if p.suffix.lower() in (".exr", ".hdr"))
        if not hdri_files:
            parser.error(f"--hdri-dir {args.hdri_dir}: no .exr / .hdr files")

    if args.mode == "index":
        uids = rebuild_training_uids(pathlib.Path(args.out))
        print(f"training_uid.json: {len(uids)} uids")
        return 0
    if args.mode == "train":
        if not args.glbs:
            parser.error("--mode train needs --glbs")
        if args.exclude is None:
            args.exclude = str(DEFAULT_EXCLUDE)
        return 1 if run_train(args, hdri_files) else 0
    if bool(args.glb) == bool(args.glb_list):
        parser.error("--mode views needs exactly one of --glb / --glb-list")
    return 1 if run_views(args, hdri_files) else 0


if __name__ == "__main__":
    code = main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    _flush_c_stdout()
    # The bpy 4.2 module segfaults at interpreter teardown once a glTF has been imported
    # (exit 139 even without rendering, measured), so skip the teardown.
    os._exit(code)
