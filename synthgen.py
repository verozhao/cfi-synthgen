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
# Scene reset & Compositor Setup
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

    # Enable Object Index pass for perfect Ground Truth masks
    bpy.context.view_layer.use_pass_object_index = True

    # Setup Compositor
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    
    # Standard output for the photorealistic RGB pass
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    
    # Dedicated File Output for the ID masks
    file_output = tree.nodes.new("CompositorNodeOutputFile")
    file_output.name = "MaskOutput"
    file_output.format.file_format = 'PNG'
    file_output.format.color_depth = '8'
    file_output.format.color_mode = 'BW'
    
    mask_dir = os.path.join(tempfile.gettempdir(), "synthgen_masks")
    os.makedirs(mask_dir, exist_ok=True)
    file_output.base_path = mask_dir
    file_output.file_slots[0].path = "mask_"
    
    # Divide the IndexOB by 255 so it saves perfectly into an 8-bit PNG channel
    div = tree.nodes.new("CompositorNodeMath")
    div.operation = 'DIVIDE'
    div.inputs[1].default_value = 255.0
    
    tree.links.new(rl.outputs["IndexOB"], div.inputs[0])
    tree.links.new(div.outputs[0], file_output.inputs[0])

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
# Rendering (RGB + Index Passes)
# ─────────────────────────────────────────────────────────────────────────

def render_rgb(path: str, resolution: int):
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

def render_mask_pass(output_name: str, resolution: int) -> np.ndarray:
    scene = bpy.context.scene
    
    # Turn off aliasing, denoising, and bounces for mathematically sharp masks
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
    
    # Read the output mask back from disk
    filepath = os.path.join(out_node.base_path, f"{output_name}_{scene.frame_current:04d}.png")
    
    with Image.open(filepath) as mask_img:
        mask_arr = np.array(mask_img)
        
    scene.view_settings.view_transform = prev_transform
    return mask_arr

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
        
        obj = import_glb_with_physics(glb_path, pos, rot)
        # Assign unique ID index (1-based, 0 is the background/ground)
        obj.pass_index = i + 1 
        objs.append(obj)

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
        stem = pathlib.Path(glb_path).parent.name
        category_ids.append(dataset.category_id(stem))

    for c in range(cameras_per_scene):
        loc, rot = sample_camera_pose(center, radius)
        set_camera(loc, rot)

        # 1. Render Photorealistic Image
        rgb_rel = f"images/{scene_idx:04d}_{c:02d}.png"
        rgb_path = os.path.join(out_dir, rgb_rel)
        render_rgb(rgb_path, resolution)

        # 2. Extract Modal (Visible) Masks all at once
        modal_arr = render_mask_pass("modal", resolution)
        modal_masks = {}
        for i, obj in enumerate(objs):
            pass_idx = i + 1
            modal_masks[i] = (modal_arr == pass_idx)

        # 3. Extract Amodal (Full) Masks individually
        amodal_masks = {}
        for i, obj in enumerate(objs):
            others = [o for j, o in enumerate(objs) if j != i]
            for o in others:
                o.hide_render = True
            
            amodal_arr = render_mask_pass(f"amodal_{i}", resolution)
            pass_idx = i + 1
            amodal_masks[i] = (amodal_arr == pass_idx)
            
            for o in others:
                o.hide_render = False

        # Add records to COCO dataset
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
    glb_files = sorted(glb_dir.glob("*/*.glb"))
    if not glb_files:
        print(f"No .glb files found in {glb_dir}")
        raise SystemExit(1)

    category_names = sorted({p.parent.name for p in glb_files})
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
    
    # Suppress normal Blender shutdown warning
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass