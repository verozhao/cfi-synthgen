"""
GLB folder -> COCO detection dataset with modal + amodal masks

Steps:
  1. import glb products and place them via a chosen strategy with rigid-body physics
  2. sample camera poses (random ring or fixed JSON config)
  3. render rgb, modal, and per-object amodal passes
  4. extract pixel-matched masks and write COCO json with metadata
"""

import argparse
import json
import math
import os
import pathlib
import random
import tempfile

import bpy
import numpy as np
from mathutils import Euler, Vector
from PIL import Image

from coco_writer import COCODataset


DEFAULT_SHAPE_SIZE_RANGES = {
    "box":         (0.10, 0.35),
    "box_rounded": (0.12, 0.30),
    "can":         (0.06, 0.20),
    "can_jar":     (0.08, 0.25),
    "bottle":      (0.12, 0.35),
    "bag":         (0.08, 0.40),
}
FALLBACK_SIZE_RANGE = (0.08, 0.35)

BOX_SHAPES = {"box", "box_rounded"}
STACKABLE_SHAPES = {"box"}  # box_rounded (soft squeeze bottles, shampoo) can't be stacked

PLACEMENT_MODES = ("scatter", "close_far", "stacking")

# CFI-3DGen GLBs are authored Y-up: +Z=front, -Z=back, +Y=top.
# Blender's glTF importer auto-rotates Y-up to Z-up. World normals come from
# obj.matrix_world.to_3x3() @ local_axis after the auto-rotation.
LOCAL_FRONT = Vector((0, 0, 1))
LOCAL_BACK  = Vector((0, 0, -1))
LOCAL_TOP   = Vector((0, 1, 0))


def random_size(shape):
    lo, hi = DEFAULT_SHAPE_SIZE_RANGES.get(shape, FALLBACK_SIZE_RANGE) if shape else FALLBACK_SIZE_RANGE
    return random.uniform(lo, hi)


def load_manifest_entry(glb_path):
    sidecar = pathlib.Path(glb_path).parent / "manifest_entry.json"
    if not sidecar.exists():
        return None
    with open(sidecar) as f:
        return json.load(f)


def resolve_sku_and_shape(glb_path):
    entry = load_manifest_entry(glb_path)
    if entry:
        return entry.get("sku") or pathlib.Path(glb_path).stem, entry.get("shape")
    return pathlib.Path(glb_path).parent.name, None


def world_aabb(obj):
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return Vector((min(xs), min(ys), min(zs))), Vector((max(xs), max(ys), max(zs)))


def aabb_overlap_volume(a_min, a_max, b_min, b_max):
    dx = max(0.0, min(a_max.x, b_max.x) - max(a_min.x, b_min.x))
    dy = max(0.0, min(a_max.y, b_max.y) - max(a_min.y, b_min.y))
    dz = max(0.0, min(a_max.z, b_max.z) - max(a_min.z, b_min.z))
    return dx * dy * dz


def any_pair_intersects(objs, tolerance_m3=1e-4, ignore_pairs=None):
    if ignore_pairs is None:
        ignore_pairs = set()
    bboxes = [world_aabb(o) for o in objs]
    n = len(objs)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in ignore_pairs or (j, i) in ignore_pairs:
                continue
            if aabb_overlap_volume(*bboxes[i], *bboxes[j]) > tolerance_m3:
                return True
    return False


def world_normals_for_obj(obj):
    rot3 = obj.matrix_world.to_3x3()
    return {
        "front_world_normal": list(rot3 @ LOCAL_FRONT),
        "back_world_normal":  list(rot3 @ LOCAL_BACK),
        "top_world_normal":   list(rot3 @ LOCAL_TOP),
    }


# ─────────────────────────────────────────────────────────────────────────
# Scene reset
# ─────────────────────────────────────────────────────────────────────────

def reset_scene(hdri_path: str | None = None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
    except Exception:
        pass
    bpy.ops.rigidbody.world_add()

    world = bpy.data.worlds.new("SynthWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg = nodes["Background"]

    if hdri_path:
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = random.uniform(0.7, 1.3)
    else:
        bg.inputs["Color"].default_value = (0.55, 0.58, 0.62, 1.0)
        bg.inputs["Strength"].default_value = 0.5

    scene.world = world
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"

    bpy.context.view_layer.use_pass_object_index = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

    file_output = tree.nodes.new("CompositorNodeOutputFile")
    file_output.name = "MaskOutput"
    file_output.format.file_format = 'PNG'
    file_output.format.color_depth = '8'
    file_output.format.color_mode = 'BW'

    mask_dir = os.path.join(tempfile.gettempdir(), "synthgen_masks")
    os.makedirs(mask_dir, exist_ok=True)
    file_output.base_path = mask_dir
    file_output.file_slots[0].path = "mask_"

    div = tree.nodes.new("CompositorNodeMath")
    div.operation = 'DIVIDE'
    div.inputs[1].default_value = 255.0

    tree.links.new(rl.outputs["IndexOB"], div.inputs[0])
    tree.links.new(div.outputs[0], file_output.inputs[0])


def build_environment(use_hdri: bool = False, bg_image_path: str | None = None):
    bpy.ops.mesh.primitive_plane_add(size=20.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    ground.rigid_body.collision_shape = "MESH"
    ground.rigid_body.friction = 1.0
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.0

    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")

    if bg_image_path and os.path.exists(bg_image_path):
        tex_node = tree.nodes.new('ShaderNodeTexImage')
        tex_node.image = bpy.data.images.load(bg_image_path, check_existing=True)
        coord_node = tree.nodes.new('ShaderNodeTexCoord')
        map_node = tree.nodes.new('ShaderNodeMapping')
        scale = random.uniform(6.0, 14.0)
        map_node.inputs['Scale'].default_value = (scale, scale, 1.0)
        tree.links.new(coord_node.outputs['UV'], map_node.inputs['Vector'])
        tree.links.new(map_node.outputs['Vector'], tex_node.inputs['Vector'])
        tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        if bsdf:
            bsdf.inputs["Roughness"].default_value = random.uniform(0.25, 0.55)
    elif bsdf:
        bsdf.inputs["Base Color"].default_value = (0.4, 0.38, 0.36, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.7

    ground.data.materials.append(mat)

    if not use_hdri:
        key = bpy.data.lights.new(name="Key", type="AREA")
        key.energy = random.uniform(120, 200)
        key.size = 3.0
        key_obj = bpy.data.objects.new("Key", key)
        key_obj.location = (
            random.uniform(1.0, 2.0),
            random.uniform(-1.6, -0.8),
            random.uniform(2.2, 3.0),
        )
        key_obj.rotation_euler = (math.radians(45), 0, math.radians(30))
        bpy.context.scene.collection.objects.link(key_obj)

        fill = bpy.data.lights.new(name="Fill", type="AREA")
        fill.energy = random.uniform(40, 80)
        fill.size = 3.0
        fill_obj = bpy.data.objects.new("Fill", fill)
        fill_obj.location = (-1.5, 1.2, 2.0)
        bpy.context.scene.collection.objects.link(fill_obj)

    return ground


# ─────────────────────────────────────────────────────────────────────────
# GLB import + physics
# ─────────────────────────────────────────────────────────────────────────

def flatten_metals(obj: bpy.types.Object):
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf and "Metallic" in bsdf.inputs:
                bsdf.inputs["Metallic"].default_value = 0.0


def _import_and_scale(glb_path, position, rotation_euler, shape, target_size=None):
    bpy.ops.import_scene.gltf(filepath=glb_path)
    selected = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not selected:
        raise RuntimeError(f"No mesh found in {glb_path}")
    if len(selected) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in selected:
            o.select_set(True)
        bpy.context.view_layer.objects.active = selected[0]
        bpy.ops.object.join()
        obj = bpy.context.active_object
    else:
        obj = selected[0]

    flatten_metals(obj)
    if target_size is None:
        target_size = random_size(shape)
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        obj.scale *= target_size / max_dim
        bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = Vector(position)
    obj.rotation_euler = Euler(rotation_euler)
    obj["target_size_m"] = float(target_size)
    obj["shape"] = shape if shape else ""
    return obj


def import_glb_with_physics(glb_path, position, rotation_euler, shape=None, target_size=None):
    obj = _import_and_scale(glb_path, position, rotation_euler, shape, target_size)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.mass = 0.1
    obj.rigid_body.friction = 0.8
    obj.rigid_body.use_margin = True
    obj.rigid_body.collision_margin = 0.001
    obj.rigid_body.linear_damping = 0.5
    obj.rigid_body.angular_damping = 0.8
    return obj


def import_glb_static(glb_path, position, rotation_euler, shape=None, target_size=None):
    return _import_and_scale(glb_path, position, rotation_euler, shape, target_size)


def bake_physics(rigid_objs, frames=120):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    if scene.rigidbody_world and scene.rigidbody_world.point_cache:
        scene.rigidbody_world.point_cache.frame_start = 1
        scene.rigidbody_world.point_cache.frame_end = frames
    bpy.ops.ptcache.bake_all(bake=True)
    scene.frame_set(frames)
    bpy.context.view_layer.update()
    for obj in rigid_objs:
        if obj.rigid_body is None:
            continue
        mat = obj.matrix_world.copy()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_remove()
        obj.matrix_world = mat
        obj.select_set(False)
    bpy.ops.ptcache.free_bake_all()


# ─────────────────────────────────────────────────────────────────────────
# Placement strategies
# ─────────────────────────────────────────────────────────────────────────

def _zone_xy(drop_bounds, scale_x=1.0, scale_y=1.0):
    return (random.uniform(drop_bounds[0] * scale_x, drop_bounds[1] * scale_x),
            random.uniform(drop_bounds[2] * scale_y, drop_bounds[3] * scale_y))


def place_scatter(glb_paths, drop_bounds):
    out = []
    for i, glb_path in enumerate(glb_paths):
        x, y = _zone_xy(drop_bounds, 1.0, 1.0)
        z = 0.25 + i * 0.15
        rot = (0.0, 0.0, random.uniform(0, 2 * math.pi))
        out.append((glb_path, (x, y, z), rot, "physics", None, None))
    return out


def place_close_far(glb_paths, drop_bounds, n_close, n_far):
    out = []
    for i, glb_path in enumerate(glb_paths):
        if i < n_close:
            x, y = _zone_xy(drop_bounds, 0.7, 0.7)
            z = 0.25 + i * 0.12
        else:
            far_dist = random.uniform(0.30, 0.50)
            far_angle = random.uniform(0, 2 * math.pi)
            x = far_dist * math.cos(far_angle)
            y = far_dist * math.sin(far_angle)
            z = 0.25 + i * 0.12
        rot = (0.0, 0.0, random.uniform(0, 2 * math.pi))
        out.append((glb_path, (x, y, z), rot, "physics", None, None))
    return out


def plan_stacking(glb_paths, glb_shapes, drop_bounds):
    """Plan a stacking scene. Returns:
        (stack_specs, side_specs)
    or None if not enough boxes.

    stack_specs: list of (glb_path, shape, layer) — to import sequentially with
                 z computed after each import (depends on real settled box height).
    side_specs:  list of (glb_path, pos, rot, "physics", None, None) — same 6-tuple
                 as scatter, ready to feed into the standard import loop.
    """
    box_indices = [i for i, s in enumerate(glb_shapes) if s in STACKABLE_SHAPES]
    other_indices = [i for i, s in enumerate(glb_shapes) if s not in STACKABLE_SHAPES]
    if len(box_indices) < 2:
        return None

    n_stack = min(len(box_indices), random.randint(2, 3))
    stack_idx = box_indices[:n_stack]
    side_idx = box_indices[n_stack:] + other_indices

    base_x, base_y = _zone_xy(drop_bounds, 0.4, 0.4)
    stack_specs = []
    for layer, idx in enumerate(stack_idx):
        stack_specs.append((glb_paths[idx], glb_shapes[idx], layer, base_x, base_y))

    side_specs = []
    n_side = len(side_idx)
    # Place side products around the stack at evenly-spaced angles (with jitter).
    angle_step = 2 * math.pi / max(n_side, 1)
    angle_offset = random.uniform(0, 2 * math.pi)
    for k, idx in enumerate(side_idx):
        side_dist = random.uniform(0.25, 0.45)
        side_angle = angle_offset + k * angle_step + random.uniform(-angle_step * 0.25, angle_step * 0.25)
        x = side_dist * math.cos(side_angle)
        y = side_dist * math.sin(side_angle)
        z = 0.30 + k * 0.20  # higher staggered drop so AABBs differ during bake
        rot = (0.0, 0.0, random.uniform(0, 2 * math.pi))
        side_specs.append((glb_paths[idx], (x, y, z), rot, "physics", None, None))

    return stack_specs, side_specs


def import_stacked_box(glb_path, shape, layer, base_x, base_y, z_floor):
    """Import a box and place it sitting on z_floor with its bottom face there.
    Returns (obj, height_used) so caller can advance z_floor for the next box.

    Key fix vs the previous version: CFI-3DGen GLBs are often authored
    horizontally (a cereal box lying on its side). We force the box upright
    by orienting its longest dimension along Z, then read the true Z extent
    after rotation to compute the next stack level.
    """
    obj = _import_and_scale(
        glb_path,
        position=(0, 0, 0),
        rotation_euler=(0, 0, 0),
        shape=shape,
        target_size=None,
    )

    dx, dy, dz = obj.dimensions
    longest = max(dx, dy, dz)
    if longest == dx:
        obj.rotation_euler = (0.0, math.radians(90), random.uniform(-math.radians(8), math.radians(8)))
    elif longest == dy:
        obj.rotation_euler = (math.radians(90), 0.0, random.uniform(-math.radians(8), math.radians(8)))
    else:
        obj.rotation_euler = (0.0, 0.0, random.uniform(-math.radians(8), math.radians(8)))
    bpy.context.view_layer.update()

    z_extent_after = obj.dimensions.z
    if hasattr(obj, "matrix_world"):
        bpy.context.view_layer.update()
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        zs = [c.z for c in corners]
        z_extent_after = max(zs) - min(zs)

    obj.location = Vector((
        base_x + random.uniform(-0.01, 0.01),
        base_y + random.uniform(-0.01, 0.01),
        z_floor + z_extent_after / 2.0,
    ))
    bpy.context.view_layer.update()

    obj["target_size_m"] = float(z_extent_after)
    obj["shape"] = shape if shape else ""

    return obj, z_extent_after


# ─────────────────────────────────────────────────────────────────────────
# Camera sampling
# ─────────────────────────────────────────────────────────────────────────

def sample_camera_pose(center, radius):
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(10), math.radians(35))
    r = radius * random.uniform(1.2, 1.8)
    x = center.x + r * math.cos(theta) * math.cos(phi)
    y = center.y + r * math.sin(theta) * math.cos(phi)
    z = center.z + r * math.sin(phi)
    location = Vector((x, y, z))
    rot = (center - location).normalized().to_track_quat("-Z", "Y").to_euler()
    return location, rot


def look_at_rotation(location, target):
    return (target - location).normalized().to_track_quat("-Z", "Y").to_euler()


def set_camera(location, rotation_euler, fov_deg=None):
    cam_data = bpy.data.cameras.get("SynthCam")
    if cam_data is None:
        cam_data = bpy.data.cameras.new("SynthCam")
    cam_obj = bpy.data.objects.get("SynthCamObj")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("SynthCamObj", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    if fov_deg is not None:
        cam_data.lens_unit = "FOV"
        cam_data.angle = math.radians(fov_deg)
    cam_obj.location = location
    cam_obj.rotation_euler = rotation_euler
    bpy.context.scene.camera = cam_obj


# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def render_rgb(path, resolution):
    scene = bpy.context.scene
    scene.cycles.samples = 64
    scene.cycles.filter_width = 1.5
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)


def render_mask_pass(output_name, resolution):
    scene = bpy.context.scene
    scene.cycles.samples = 1
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    prev_transform = scene.view_settings.view_transform
    scene.view_settings.view_transform = "Raw"
    out_node = scene.node_tree.nodes["MaskOutput"]
    out_node.file_slots[0].path = f"{output_name}_"
    bpy.ops.render.render(write_still=False)
    filepath = os.path.join(out_node.base_path, f"{output_name}_{scene.frame_current:04d}.png")
    with Image.open(filepath) as mask_img:
        mask_arr = np.array(mask_img)
    scene.view_settings.view_transform = prev_transform
    return mask_arr


# ─────────────────────────────────────────────────────────────────────────
# Scene loop
# ─────────────────────────────────────────────────────────────────────────

def generate_scene(glb_paths, scene_idx, cameras, resolution, out_dir, dataset,
                   placement_mode, n_close, n_far,
                   hdri_path=None, bg_image_path=None, drop_bounds=None,
                   max_intersect_retries=3):
    reset_scene(hdri_path=hdri_path)
    build_environment(use_hdri=bool(hdri_path), bg_image_path=bg_image_path)

    if drop_bounds is None:
        drop_bounds = (-0.12, 0.12, -0.12, 0.12)

    skus = []
    shapes = []
    for glb_path in glb_paths:
        sku, shape = resolve_sku_and_shape(glb_path)
        skus.append(sku)
        shapes.append(shape)

    actual_mode = placement_mode
    stack_member_indices = []
    stack_specs = None
    side_specs = None

    if placement_mode == "stacking":
        result = plan_stacking(glb_paths, shapes, drop_bounds)
        if result is None:
            print(f"  [scene {scene_idx}] not enough box products for stacking, falling back to scatter")
            placements = place_scatter(glb_paths, drop_bounds)
            actual_mode = "scatter"
        else:
            stack_specs, side_specs = result
            placements = side_specs
    elif placement_mode == "close_far":
        placements = place_close_far(glb_paths, drop_bounds, n_close, n_far)
    else:
        placements = place_scatter(glb_paths, drop_bounds)

    glb_path_to_meta = {p: (skus[i], shapes[i]) for i, p in enumerate(glb_paths)}

    objs = []
    physics_objs = []
    obj_skus = []
    obj_shapes = []
    obj_layers = []

    if stack_specs is not None:
        z_floor = 0.0
        for (glb_path, shape, layer, bx, by) in stack_specs:
            obj, h = import_stacked_box(glb_path, shape, layer, bx, by, z_floor)
            sku, _ = glb_path_to_meta[glb_path]
            obj.pass_index = len(objs) + 1
            objs.append(obj)
            obj_skus.append(sku)
            obj_shapes.append(shape)
            obj_layers.append(layer)
            stack_member_indices.append(len(objs) - 1)
            z_floor += h + 0.001

    for (glb_path, pos, rot, mode_tag, layer, target_size) in placements:
        sku, shape = glb_path_to_meta[glb_path]
        if mode_tag == "static":
            obj = import_glb_static(glb_path, pos, rot, shape=shape, target_size=target_size)
        else:
            obj = import_glb_with_physics(glb_path, pos, rot, shape=shape, target_size=target_size)
            physics_objs.append(obj)
        obj.pass_index = len(objs) + 1
        objs.append(obj)
        obj_skus.append(sku)
        obj_shapes.append(shape)
        obj_layers.append(layer)

    if physics_objs:
        bake_physics(physics_objs, frames=120)

    ignore_pairs = set()
    for a in stack_member_indices:
        for b in stack_member_indices:
            if a < b:
                ignore_pairs.add((a, b))

    if any_pair_intersects(objs, ignore_pairs=ignore_pairs):
        print(f"  [scene {scene_idx}] AABB intersection detected; retrying ({max_intersect_retries-1} left)")
        for o in list(objs):
            bpy.data.objects.remove(o, do_unlink=True)
        if max_intersect_retries > 1:
            return generate_scene(glb_paths, scene_idx, cameras, resolution, out_dir, dataset,
                                  placement_mode, n_close, n_far, hdri_path, bg_image_path,
                                  drop_bounds, max_intersect_retries - 1)
        print(f"  [scene {scene_idx}] giving up after retries; skipping")
        return False

    locations = [o.location for o in objs]
    center = sum(locations, Vector((0, 0, 0))) / len(locations)
    max_dist = 0.05
    for o in objs:
        for corner in o.bound_box:
            world_corner = o.matrix_world @ Vector(corner)
            dist = (world_corner - center).length
            if dist > max_dist:
                max_dist = dist
    radius = max_dist

    category_ids = [dataset.category_id(s) for s in obj_skus]

    for c, cam_spec in enumerate(cameras):
        if cam_spec is None:
            loc, rot = sample_camera_pose(center, radius)
            fov_deg = None
        else:
            loc = Vector(cam_spec["location"])
            if "look_at" in cam_spec:
                rot = look_at_rotation(loc, Vector(cam_spec["look_at"]))
            elif "rotation_euler_deg" in cam_spec:
                rot = Euler([math.radians(d) for d in cam_spec["rotation_euler_deg"]])
            else:
                raise ValueError("camera spec needs 'look_at' or 'rotation_euler_deg'")
            fov_deg = cam_spec.get("fov_deg")
        set_camera(loc, rot, fov_deg)

        rgb_rel = f"images/{scene_idx:04d}_{c:02d}.png"
        rgb_path = os.path.join(out_dir, rgb_rel)
        render_rgb(rgb_path, resolution)

        modal_arr = render_mask_pass("modal", resolution)
        modal_masks = {i: (modal_arr == (i + 1)) for i in range(len(objs))}

        amodal_masks = {}
        for i, obj in enumerate(objs):
            others = [o for j, o in enumerate(objs) if j != i]
            for o in others:
                o.hide_render = True
            amodal_arr = render_mask_pass(f"amodal_{i}", resolution)
            amodal_masks[i] = (amodal_arr == (i + 1))
            for o in others:
                o.hide_render = False

        image_id = dataset.add_image(rgb_rel, resolution, resolution)
        cam_distance_to_center = (Vector(loc) - center).length

        for i, obj in enumerate(objs):
            if modal_masks[i].sum() == 0:
                continue
            ann_meta = {
                "shape": obj.get("shape") or None,
                "target_size_m": float(obj.get("target_size_m", 0.0)),
                "placement_mode": actual_mode,
                "stack_layer": obj_layers[i],
                "camera_distance_m": float(cam_distance_to_center),
                **world_normals_for_obj(obj),
            }
            dataset.add_annotation(image_id, category_ids[i],
                                   modal_masks[i], amodal_masks[i],
                                   metadata=ann_meta)
    return True


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset from GLBs")
    parser.add_argument("--glbs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--products-per-scene", type=int, default=6)
    parser.add_argument("--cameras-per-scene", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hdri", default=None)
    parser.add_argument("--cameras-json", default=None)
    parser.add_argument("--backgrounds", default=None)
    parser.add_argument("--placement", choices=PLACEMENT_MODES, default="scatter",
                        help="Placement strategy. stacking requires >=2 box products in pool.")
    parser.add_argument("--n-close", type=int, default=3,
                        help="Products near scene center in close_far mode.")
    parser.add_argument("--n-far", type=int, default=1,
                        help="Products offset away in close_far mode.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    glb_dir = pathlib.Path(args.glbs)
    glb_files = sorted(glb_dir.rglob("*.glb"))
    if not glb_files:
        print(f"No .glb files found in {glb_dir}")
        raise SystemExit(1)

    if args.placement == "stacking":
        box_count = sum(1 for p in glb_files if resolve_sku_and_shape(p)[1] in STACKABLE_SHAPES)
        if box_count < 2:
            print(f"Stacking mode requires >=2 box products in pool; only {box_count} found.")
            raise SystemExit(1)

    if args.placement == "close_far":
        if args.n_close + args.n_far != args.products_per_scene:
            args.products_per_scene = args.n_close + args.n_far
            print(f"close_far: setting products_per_scene = {args.products_per_scene}")

    category_names = sorted({resolve_sku_and_shape(p)[0] for p in glb_files})
    n_with_manifest = sum(1 for p in glb_files if (p.parent / "manifest_entry.json").exists())
    print(f"Found {len(glb_files)} GLBs ({n_with_manifest} with manifest_entry.json)")
    print(f"COCO categories: {len(category_names)}")
    print(f"Placement mode: {args.placement}")

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    dataset = COCODataset(category_names)

    if args.cameras_json:
        with open(args.cameras_json) as f:
            cam_data = json.load(f)
        if isinstance(cam_data, dict):
            fixed_cameras = cam_data["cameras"]
            dz = cam_data.get("drop_zone", [0.12, 0.12])
            drop_bounds = (-dz[0], dz[0], -dz[1], dz[1])
        else:
            fixed_cameras = cam_data
            drop_bounds = None
    else:
        fixed_cameras = None
        drop_bounds = None

    bg_files = []
    if args.backgrounds:
        bg_dir = pathlib.Path(args.backgrounds)
        bg_files = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) + list(bg_dir.glob("*.jpeg"))
        if not bg_files:
            print(f"Warning: No images found in {args.backgrounds}")

    successes = 0
    for scene_idx in range(args.scenes):
        k = args.products_per_scene
        if args.placement == "stacking":
            box_pool = [p for p in glb_files if resolve_sku_and_shape(p)[1] in STACKABLE_SHAPES]
            non_box_pool = [p for p in glb_files if resolve_sku_and_shape(p)[1] not in STACKABLE_SHAPES]
            n_box = min(len(box_pool), random.randint(2, 3))
            n_other = max(0, k - n_box)
            chosen_boxes = random.sample(box_pool, n_box)
            chosen_others = random.sample(non_box_pool, min(n_other, len(non_box_pool)))
            chosen = chosen_boxes + chosen_others
        elif len(glb_files) >= k:
            chosen = random.sample(glb_files, k)
        else:
            chosen = random.choices(glb_files, k=k)

        glb_paths = [str(p) for p in chosen]
        cameras = fixed_cameras if fixed_cameras is not None else [None] * args.cameras_per_scene
        chosen_bg = str(random.choice(bg_files)) if bg_files else None

        ok = generate_scene(glb_paths, scene_idx, cameras,
                            args.resolution, args.out, dataset,
                            placement_mode=args.placement,
                            n_close=args.n_close, n_far=args.n_far,
                            hdri_path=args.hdri, bg_image_path=chosen_bg,
                            drop_bounds=drop_bounds)
        if ok:
            successes += 1
        print(f"Scene {scene_idx + 1}/{args.scenes} {'done' if ok else 'SKIPPED'}")

    ann_path = os.path.join(args.out, "annotations.json")
    dataset.save(ann_path)
    print(f"Saved {ann_path}  ({successes}/{args.scenes} scenes succeeded)")

    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass
