import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from unitex import walkthrough_figures as F

NS = "{http://www.w3.org/2000/svg}"


def _fig(tmp_path, glyphs=("charms", "18.6", "STRAWBERRY")):
    fig = tmp_path / "fig"
    fig.mkdir()
    for n in ("photo.jpg", "rembg.jpg", "geometry.jpg", "texture_pass.jpg", "delight.jpg", "photo_front.jpg", "uv.jpg"):
        Image.new("RGB", (40, 30), (200, 0, 0)).save(fig / n)
    for n in ("mesh_34.png", "final_34.png", "points.png"):
        Image.new("RGBA", (30, 30), (0, 0, 0, 0)).save(fig / n)
    json.dump({"sku": "1", "run": "r", "view_res": 1024, "uv_res": 2048, "faces": 40000, "points": 32768,
               "glyphs": list(glyphs)}, open(fig / "info.json", "w"))
    return fig


def test_compare_figure_is_plain_svg_with_three_paid_steps():
    svg = F.compare_svg()
    ET.fromstring(svg)                                                     # well-formed
    for old in ("#e09a2d", "#fff3e0", "#fdece6", "#b4441c"):              # amber and salmon of the first version
        assert old not in svg
    assert svg.count(">Paid API call") == 3 and "Paid API call, the main cost" in svg
    assert svg.count(f'stroke="{F.ACC}"') == 3                            # two boxes of ours + the legend


def test_overview_has_ten_stages_with_embedded_outputs(tmp_path):
    svg = F.overview_svg(_fig(tmp_path), "Lucky Charms", ["STRAWBERRY", "NOT IN THE RUN", "18.6"])
    root = ET.fromstring(svg)
    texts = [t.text for t in root.iter(NS + "text")]
    assert [t for t in texts if t and t.isdigit()] == [str(i) for i in range(1, 11)]
    assert texts.count("OURS") == 2
    assert "STRAWBERRY" in texts and "18.6" in texts                      # chosen glyphs, in that order
    assert texts.index("STRAWBERRY") < texts.index("18.6")
    assert "NOT IN THE RUN" not in texts and "charms" not in texts
    hrefs = [im.get("href") for im in root.iter(NS + "image")]
    assert len(hrefs) == 10 and all(h.startswith("data:image/") for h in hrefs)
    assert "(40k triangles, from ShapeGen)." in texts and "32,768 surface points" in texts
    assert "2048 px UV texture." in texts and "(Lucky Charms), final run at 1024 px" in svg


def test_figures_go_on_top_once_and_a_rerun_replaces_them():
    page = ('<html><body><header><nav><a href="#p1">P1</a></nav></header><main><h2 id="p1">P1</h2></main>'
            '</body></html>')
    doc = F.add_figures(page, "<svg>A</svg>", "<svg>B</svg>", "X & Y")
    assert doc.index('id="overview"') < doc.index('id="compare"') < doc.index('id="p1"')
    assert '<nav><a href="#overview">Pipeline overview</a><a href="#compare">Pre-summer vs UniTEX</a><a href="#p1">' in doc
    assert "for X &amp; Y (our final run)" in doc
    again = F.add_figures(doc, "<svg>C</svg>", "<svg>D</svg>", "X & Y")
    assert again == F.add_figures(page, "<svg>C</svg>", "<svg>D</svg>", "X & Y")


def test_page_keeps_the_input_page(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<nav></nav><main></main>")
    with pytest.raises(SystemExit):
        F.main(["page", "--fig", str(_fig(tmp_path)), "--page", str(page), "--name", "X", "--out", str(page)])
    out = tmp_path / "out.html"
    F.main(["page", "--fig", str(tmp_path / "fig"), "--page", str(page), "--name", "X", "--out", str(out),
            "--svg-dir", str(tmp_path / "svg")])
    assert page.read_text() == "<nav></nav><main></main>"
    assert 'id="overview"' in out.read_text() and len(list((tmp_path / "svg").glob("*.svg"))) == 2


def test_render_shades_the_three_visible_faces_of_a_box():
    trimesh = pytest.importorskip("trimesh")
    im = F.render(trimesh.creation.box(extents=(1, 2, 0.6)), res=96, textured=False)
    a = np.asarray(im)
    assert im.size == (96, 96) and (a[..., 3] > 0).mean() > 0.25
    assert len({tuple(p) for p in a[a[..., 3] > 0][:, :3]}) == 3          # front, side and top, flat shaded


def test_assets_from_a_small_run_cache(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    R = 16
    d = tmp_path / "eval" / "123"
    c = d / "ours" / "cache"
    (c / "w_LTM").mkdir(parents=True)
    (c / "wo_LTM").mkdir()
    trimesh.creation.box().export(d / "mesh.glb")
    trimesh.creation.box().export(c / "w_LTM" / "textured_mesh.glb")
    photo = Image.new("RGB", (64, 64), (255, 255, 255))
    photo.paste((200, 0, 0), (16, 8, 48, 56))
    photo.save(d / "ref.png")
    photo.save(c / "rembg_image.png")
    grid = Image.new("RGB", (3 * R, 2 * R), (128, 128, 128))
    grid.paste((200, 0, 0), (4, 4, 12, 12))
    for n in ("mv_normal.png", "mv_ccm.png", "mv_rgb.png"):
        grid.save(c / n)
    Image.new("RGB", (6 * R, R), (0, 150, 0)).save(c / "mv_rgb_w_light.png")
    Image.new("RGB", (32, 32), (10, 10, 10)).save(c / "wo_LTM" / "completed_uv.png")
    trimesh.PointCloud(np.random.default_rng(0).random((200, 3))).export(c / "coarse_pcd_fps.ply")
    json.dump({"instances": [{"text": "18.6"}, {"text": "OZ"}], "n_instances": 2}, open(c / "glyph_tokens.json", "w"))
    info = F.make_assets(tmp_path / "eval", "123", "ours", tmp_path / "fig")
    assert info == {"sku": "123", "run": "ours", "view_res": R, "uv_res": 32, "faces": 12, "points": 200,
                    "glyphs": ["18.6", "OZ"]}
    assert Image.open(tmp_path / "fig" / "photo_front.jpg").size == (8, 8)     # front tile cropped to its product
                                                                                # (the 4% margin is 0 px at 16 px)
    svg = F.overview_svg(tmp_path / "fig", "Box")                         # everything the figure reads is there
    assert ">18.6<" in svg and ">OZ<" in svg
