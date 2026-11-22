"""
Shorter versions of UniTEX's bake step (CustomRGBTextureFullPipeline.step_2_ablition) for run_unitex_fast.py and
photo_front.py.

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
