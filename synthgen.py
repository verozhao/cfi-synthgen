"""
GLB folder -> COCO detection dataset with modal + amodal masks

Steps:
  1. import glb products and drop them onto a plane with rigid-body physics
  2. sample random camera poses around the settled scene
  3. render rgb, flat-color modal, and per-object amodal passes
  4. extract pixel-matched masks and write COCO json
"""

import argparse
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

# ─────────────────────────────────────────────────────────────────────────
# Scene reset
# ─────────────────────────────────────────────────────────────────────────

def reset_scene():
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
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    bg.inputs["Strength"].default_value = 0.1
    scene.world = world
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -1.0

# ─────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────

def build_environment() -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    bpy.ops.rigidbody.object_add(type="PASSIVE")
    ground.rigid_body.collision_shape = "MESH"
    ground.rigid_body.friction = 1.0

    mat = bpy.data.materials.new(name="GroundMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    
    if bsdf:
        tex_checker = mat.node_tree.nodes.new("ShaderNodeTexChecker")
        tex_checker.inputs["Color1"].default_value = (0.15, 0.14, 0.13, 1.0)
        tex_checker.inputs["Color2"].default_value = (0.10, 0.09, 0.08, 1.0)
        tex_checker.inputs["Scale"].default_value = 25.0
        mat.node_tree.links.new(tex_checker.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.85
        
    ground.data.materials.append(mat)

    key = bpy.data.lights.new(name="Key", type="AREA")
    key.energy = 150
    key.size = 3.0
    key_obj = bpy.data.objects.new("Key", key)
    key_obj.location = (1.5, -1.2, 2.5)
    key_obj.rotation_euler = (math.radians(45), 0, math.radians(30))
    bpy.context.scene.collection.objects.link(key_obj)

    fill = bpy.data.lights.new(name="Fill", type="AREA")
    fill.energy = 60
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


def import_glb_with_physics(glb_path: str, position: tuple, rotation_euler: tuple) -> bpy.types.Object:
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

    max_dim = max(obj.dimensions)
    if max_dim > 0:
        scale_factor = 0.3 / max_dim
        obj.scale *= scale_factor
        bpy.ops.object.transform_apply(scale=True)

    obj.location = Vector(position)
    obj.rotation_euler = Euler(rotation_euler)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add(type="ACTIVE")
    obj.rigid_body.collision_shape = "CONVEX_HULL"
    obj.rigid_body.mass = 0.1
    obj.rigid_body.friction = 0.8

    return obj

# ─────────────────────────────────────────────────────────────────────────
# Physics bake
# ─────────────────────────────────────────────────────────────────────────

def bake_physics(rigid_objs: list[bpy.types.Object], frames: int = 80):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames

    for f in range(1, frames + 1):
        scene.frame_set(f)

    for obj in rigid_objs:
        mat = obj.matrix_world.copy()
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_remove()
        obj.matrix_world = mat
        obj.select_set(False)

# ─────────────────────────────────────────────────────────────────────────
# Camera sampling
# ─────────────────────────────────────────────────────────────────────────

def sample_camera_pose(center: Vector, radius: float) -> tuple[Vector, Euler]:
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.radians(15), math.radians(55))
    r = radius * random.uniform(2.5, 4.0)

    x = center.x + r * math.cos(theta) * math.cos(phi)
    y = center.y + r * math.sin(theta) * math.cos(phi)
    z = center.z + r * math.sin(phi)
    location = Vector((x, y, z))

    direction = (center - location).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    euler = rot_quat.to_euler()
    return location, euler


def set_camera(location: Vector, rotation_euler: Euler):
    cam_data = bpy.data.cameras.get("SynthCam")
    if cam_data is None:
        cam_data = bpy.data.cameras.new("SynthCam")
    cam_obj = bpy.data.objects.get("SynthCamObj")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("SynthCamObj", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = location
    cam_obj.rotation_euler = rotation_euler
    bpy.context.scene.camera = cam_obj

# ─────────────────────────────────────────────────────────────────────────
# Flat-color materials
# ─────────────────────────────────────────────────────────────────────────

def slot_color(i: int) -> tuple[int, int, int]:
    r = ((i + 1) * 37) % 256
    g = ((i + 1) * 73) % 256
    b = ((i + 1) * 131) % 256
    if r == 0 and g == 0 and b == 0:
        r = 1
    return (r, g, b)


def assign_flat_material(obj: bpy.types.Object, rgb: tuple[float, float, float]):
    obj["_orig_mats"] = [slot.material for slot in obj.material_slots]
    
    # Restrict visibility so objects don't cast shadows or reflect light onto each other
    obj["_orig_vis_diffuse"] = obj.visible_diffuse
    obj["_orig_vis_glossy"] = obj.visible_glossy
    obj["_orig_vis_shadow"] = obj.visible_shadow
    obj.visible_diffuse = False
    obj.visible_glossy = False
    obj.visible_shadow = False

    mat = bpy.data.materials.new(name=f"Flat_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*rgb, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def restore_materials(obj: bpy.types.Object):
    orig = obj.get("_orig_mats", [])
    obj.data.materials.clear()
    for mat in orig:
        obj.data.materials.append(mat)
    if "_orig_mats" in obj:
        del obj["_orig_mats"]

    if "_orig_vis_diffuse" in obj:
        obj.visible_diffuse = obj.pop("_orig_vis_diffuse")
    if "_orig_vis_glossy" in obj:
        obj.visible_glossy = obj.pop("_orig_vis_glossy")
    if "_orig_vis_shadow" in obj:
        obj.visible_shadow = obj.pop("_orig_vis_shadow")

# ─────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────

def render_rgb(path: str, resolution: int):
    scene = bpy.context.scene
    scene.cycles.samples = 64
    scene.cycles.filter_width = 1.5
    scene.cycles.use_denoising = True  # Restore denoising for photorealism
    scene.cycles.max_bounces = 12      # Restore normal lighting bounces
    
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def render_flat_to_buffer(resolution: int) -> np.ndarray:
    scene = bpy.context.scene
    scene.cycles.samples = 1
    scene.cycles.filter_width = 0.01
    
    # Disable denoising and light bounces so the flat emission colors remain mathematically perfect
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    prev_transform = scene.view_settings.view_transform
    prev_look = scene.view_settings.look
    prev_exp = scene.view_settings.exposure
    
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0

    world = scene.world
    bg = world.node_tree.nodes["Background"]
    prev_color = tuple(bg.inputs["Color"].default_value)
    prev_strength = bg.inputs["Strength"].default_value
    bg.inputs["Color"].default_value = (0, 0, 0, 1)
    bg.inputs["Strength"].default_value = 1.0

    bpy.ops.render.render()

    bg.inputs["Color"].default_value = prev_color
    bg.inputs["Strength"].default_value = prev_strength
    
    scene.view_settings.view_transform = prev_transform
    scene.view_settings.look = prev_look
    scene.view_settings.exposure = prev_exp

    result = bpy.data.images["Render Result"]
    w, h = result.size
    pixels = np.array(result.pixels[:]).reshape(h, w, 4)
    return np.flipud(pixels)

# ─────────────────────────────────────────────────────────────────────────
# Mask extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_modal_masks(flat_rgba: np.ndarray, id_to_rgb_norm: dict[int, tuple[float, float, float]]) -> dict[int, np.ndarray]:
    rgb = flat_rgba[:, :, :3]
    masks = {}
    for obj_id, expected in id_to_rgb_norm.items():
        diff = np.abs(rgb - np.array(expected))
        # Increased tolerance slightly to account for floating point precision in linear space
        masks[obj_id] = np.all(diff < 0.05, axis=-1)
    return masks


def render_amodal_mask(obj: bpy.types.Object, other_objs: list[bpy.types.Object],
                       ground: bpy.types.Object, resolution: int) -> np.ndarray:
    hidden_states = {}
    for o in other_objs + [ground]:
        hidden_states[o] = o.hide_render
        o.hide_render = True

    flat_rgba = render_flat_to_buffer(resolution)

    for o, state in hidden_states.items():
        o.hide_render = state

    rgb = flat_rgba[:, :, :3]
    return np.any(rgb > 0.02, axis=-1)

# ─────────────────────────────────────────────────────────────────────────
# Scene loop
# ─────────────────────────────────────────────────────────────────────────

def generate_scene(glb_paths: list[str], scene_idx: int, cameras_per_scene: int,
                   resolution: int, out_dir: str, dataset: COCODataset):
    reset_scene()
    ground = build_environment()

    objs = []
    for i, glb_path in enumerate(glb_paths):
        pos = (random.uniform(-0.12, 0.12), random.uniform(-0.12, 0.12), 0.25 + i * 0.15)
        rot = (random.uniform(0, 2 * math.pi),
               random.uniform(0, 2 * math.pi),
               random.uniform(0, 2 * math.pi))
        objs.append(import_glb_with_physics(glb_path, pos, rot))

    bake_physics(objs, frames=120)

    locations = [o.location for o in objs]
    center = sum(locations, Vector((0, 0, 0))) / len(locations)

    max_dist = 0.3
    for o in objs:
        for corner in o.bound_box:
            world_corner = o.matrix_world @ Vector(corner)
            dist = (world_corner - center).length
            if dist > max_dist:
                max_dist = dist
    radius = max_dist

    category_ids = []
    for glb_path in glb_paths:
        stem = pathlib.Path(glb_path).stem
        category_ids.append(dataset.category_id(stem))

    light_objs = [o for o in bpy.data.objects if o.type == "LIGHT"]

    for c in range(cameras_per_scene):
        loc, rot = sample_camera_pose(center, radius)
        set_camera(loc, rot)

        rgb_rel = f"images/{scene_idx:04d}_{c:02d}.png"
        rgb_path = os.path.join(out_dir, rgb_rel)
        render_rgb(rgb_path, resolution)

        id_to_rgb_norm = {}
        for i, obj in enumerate(objs):
            rgb_255 = slot_color(i)
            rgb_norm = (rgb_255[0] / 255.0, rgb_255[1] / 255.0, rgb_255[2] / 255.0)
            assign_flat_material(obj, rgb_norm)
            id_to_rgb_norm[i] = rgb_norm

        ground.hide_render = True
        for l in light_objs:
            l.hide_render = True

        flat_rgba = render_flat_to_buffer(resolution)
        modal_masks = extract_modal_masks(flat_rgba, id_to_rgb_norm)

        amodal_masks = {}
        for i, obj in enumerate(objs):
            others = [o for j, o in enumerate(objs) if j != i]
            amodal_masks[i] = render_amodal_mask(obj, others, ground, resolution)

        ground.hide_render = False
        for l in light_objs:
            l.hide_render = False
        for obj in objs:
            restore_materials(obj)

        image_id = dataset.add_image(rgb_rel, resolution, resolution)

        for i, obj in enumerate(objs):
            if modal_masks[i].sum() == 0:
                continue
            dataset.add_annotation(image_id, category_ids[i], modal_masks[i], amodal_masks[i])

# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic COCO dataset from GLBs")
    parser.add_argument("--glbs", required=True, help="Directory containing .glb files")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--scenes", type=int, default=100)
    parser.add_argument("--products-per-scene", type=int, default=3)
    parser.add_argument("--cameras-per-scene", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    glb_dir = pathlib.Path(args.glbs)
    glb_files = sorted(glb_dir.glob("*.glb"))
    if not glb_files:
        print(f"No .glb files found in {glb_dir}")
        raise SystemExit(1)

    category_names = sorted(p.stem for p in glb_files)
    glb_by_name = {p.stem: str(p) for p in glb_files}

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    dataset = COCODataset(category_names)

    for scene_idx in range(args.scenes):
        k = args.products_per_scene
        pool = list(glb_files)
        if len(pool) >= k:
            chosen = random.sample(pool, k)
        else:
            chosen = random.choices(pool, k=k)
        glb_paths = [str(p) for p in chosen]
        generate_scene(glb_paths, scene_idx, args.cameras_per_scene,
                       args.resolution, args.out, dataset)
        print(f"Scene {scene_idx + 1}/{args.scenes} done")

    ann_path = os.path.join(args.out, "annotations.json")
    dataset.save(ann_path)
    print(f"Saved {ann_path}")