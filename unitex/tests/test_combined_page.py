import json
import re

from PIL import Image

from unitex import combined_page as P


def _png(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


def test_page_is_self_contained_with_every_model_and_scene(tmp_path):
    views, scenes, titles = tmp_path / "views", tmp_path / "scenes", tmp_path / "titles"
    skus = ["016000233164", "111111111111"]
    for sku in set(skus) | set(P.PROGRESS_SKUS):
        _png(views / "photos" / f"{sku}.png", (200, 0, 0))
        (titles / sku).mkdir(parents=True, exist_ok=True)
        json.dump({"title": f"Product {sku}"}, open(titles / sku / "manifest_entry.json", "w"))
        for key, _ in P.MODELS + P.PROGRESS:
            for v in "0312":
                if key == P.MODELS[-1][0] and sku not in skus:
                    continue          # the per-product SKU list comes from the last model's views
                _png(views / key / sku / f"view_0{v}.png", (0, 120, 0))
    for key, _ in P.SCENE_SETS:
        for p in P.PLACEMENTS:
            for n in ("0000_00.png", "0000_01.png"):
                _png(scenes / key / p / "images" / n, (0, 0, 160))
    out = tmp_path / "page.html"
    P.main(["--views", str(views), "--scenes", str(scenes), "--titles", str(titles), "--out", str(out)])
    doc = out.read_text()
    assert not re.findall(r'(?:src|href)="(?!data:|#)', doc)            # nothing outside the file
    assert 'class="missing"' not in doc                                  # every image found
    for _, label in P.MODELS + P.SCENE_SETS:
        assert label in doc
    n_img = doc.count('src="data:image/jpeg;base64,')
    per_product = len(skus) * (1 + len(P.MODELS) * 3)
    progress = len(P.PROGRESS_SKUS) * (1 + len(P.PROGRESS))
    scene_imgs = len(P.PLACEMENTS) * 2 * len(P.SCENE_SETS)
    assert n_img == per_product + progress + scene_imgs


def test_hires_page_crops_to_the_product_and_embeds_webp(tmp_path):
    import base64
    import io

    import pytest
    from PIL import features
    if not features.check("webp"):
        pytest.skip("Pillow without WebP")
    views, hr, scenes, titles = (tmp_path / n for n in ("views", "hr", "scenes", "titles"))
    sku = "016000233164"
    for s in set([sku]) | set(P.PROGRESS_SKUS):
        _png(views / "photos" / f"{s}.png", (255, 255, 255))
        for key, _ in P.MODELS + P.PROGRESS:
            for v in "0312":
                _png(views / key / s / f"view_0{v}.png", (0, 120, 0))
            im = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
            im.paste((200, 30, 30, 255), (150, 100, 250, 300))           # 100 x 200 product
            (hr / key / s).mkdir(parents=True, exist_ok=True)
            im.save(hr / key / s / "view_00.png")
    for key, _ in P.SCENE_SETS:
        for p in P.PLACEMENTS:
            _png(scenes / key / p / "images" / "0000_00.png", (0, 0, 160))
    out = tmp_path / "hires.html"
    P.main(["--views", str(views), "--scenes", str(scenes), "--titles", str(titles), "--out", str(out),
            "--hires-views", str(hr)])
    doc = out.read_text()
    assert not re.findall(r'(?:src|href)="(?!data:|#)', doc)
    assert 'class="missing"' not in doc
    webps = re.findall(r'src="data:image/webp;base64,([^"]+)"', doc)
    assert webps
    sizes = {Image.open(io.BytesIO(base64.b64decode(w))).size for w in webps}
    assert (124, 224) in sizes      # the 100 x 200 product plus a 12 px margin (3% of the 400 px canvas)
