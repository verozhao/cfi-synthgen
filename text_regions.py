"""
UV-space OCR -> per-uid text annotations for UniTEX-FLUX training (text.json, source "uv-ocr")

Steps per sku (a render/<volume>/<sku>/ dir written by mvgen.py --mode train):
  1. extract the GLB's baseColor texture (metadata.json source_glb, then the glTF JSON + BIN chunk:
     texture -> image -> bufferView chain, stdlib only)
  2. OCR the texture with unitex.ocr (rotations 0/90/270 so vertical side-panel text is found),
     and every line becomes an item whose src_quad is its texture-pixel quad in reading order
  3. flags (template_leak, low_conf, rotated) and, with --bundle-v2, provenance from CFI-3DGen's
     template_meta.json islands (canvas px scaled to the texture): "front" inside the front
     island (for cans the FRONT quarter of the body strip), else "generated"
  4. rasterize every src_quad into an instance-id map in texture pixels
  5. per raw view: texture pixel of every foreground pixel (000i_uv.npz), its instance id,
     its reading-frame (s, t) by inverse bilinear mapping inside the item's quad, then per item
     bbox, quad, pixels, coverage, cos, height_px and the 16 x 4 (s, t) grid
  6. --verify-view-ocr: OCR a rectified crop of every view quad from 000i_albedo.png and store
     view_ocr_text / view_ocr_ned, so training can keep only boxes whose rendered text matches

Geometry per item and view: pixels are binned into the item's (s, t) cells and each cell gets a
local affine map xy = mean + J (st - mean_st) fitted over its 3 x 3 cell neighbourhood (the
item-wide fit where that is degenerate). Grid points are those maps at the cell centres, quad
corners are the nearest cell's map at the corners of the visible (s, t) range, height_px is the
median |dxy/dt|. So curved and foreshortened labels get correct points, not pixel-mean biased ones.

Writes render/<volume>/<sku>/text.json (format: unitex/DESIGN.md) plus the extra keys "island"
per item, "view_ocr_text" / "view_ocr_ned" per view (with --verify-view-ocr) and top-level
"texture", "ocr", "template", "uv_check", "stats". Existing files are skipped unless --overwrite.
Errors are logged per sku to <root>/text_regions_errors.log and the run continues.

Usage (testenv or any env with numpy + pillow, no bpy):
  python text_regions.py --root OUT --bundle-v2 /Users/test/CFI-3DGen/approved_bundle_v2 \
      --backend vision --verify-view-ocr --debug-dir OUT/text_debug
  python text_regions.py --root OUT --backend paddle --skus 016000263192,611269818994 --overwrite
"""

import argparse
import base64
import colorsys
import io
import json
import math
import os
import pathlib
import re
import struct
import sys
import time
import traceback
from dataclasses import dataclass, field
from urllib.parse import unquote

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unitex import ocr as ocr_mod
from unitex.common import STRIP_NAMES, raw_to_slot, stack_strip
from unitex.glyph import GlyphConfig, crop_gt
from unitex.text_metrics import ned, normalize

NS, NT = 16, 4                      # grid cells along the baseline / across the text
TEXT_JSON_VERSION = 1

TEMPLATE_WORDS = ("LEFT", "RIGHT", "FRONT", "BACK", "TOP", "BOTTOM")
# whole word, but "_" and "-" count as separators so the bottle pads' RIGHT_PAD / LEFT_PAD match
TEMPLATE_LEAK_RE = re.compile(r"(?<![^\W_])(" + "|".join(TEMPLATE_WORDS) + r")(?![^\W_])", re.I)
# OCR sometimes spells Latin capitals with Cyrillic / Greek look-alikes (Vision read "BоTтоM")
HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХаеокмнорстхΑΒΕΚΜΝΟΡΤΧο", "ABEKMHOPCTXaeokmhopctxABEKMNOPTXo")
LOW_CONF = 0.5
ROTATED_TOL_DEG = 10.0

# CFI-3DGen can unwrap (uv6_v2.py:350 BODY_SECTION_LABELS): the body strip is 4 equal sections,
# left to right. Its prompt_hint lists another order, but the texture follows the drawn labels
# (Red Bull 611269818994: wordmark at 0.28-0.49 of the strip width).
CAN_BODY_SECTIONS = ("LEFT", "FRONT", "RIGHT", "BACK")

GLB_MAGIC = b"glTF"
CHUNK_JSON, CHUNK_BIN = 0x4E4F534A, 0x004E4942
WRAP_REPEAT, WRAP_CLAMP, WRAP_MIRROR = 10497, 33071, 33648

MIN_PIXELS = 8
MIN_HEIGHT_PX = 1.0
MIN_ST_VAR = 1e-4                   # item-wide (s, t) variance floor (std 0.01) for the affine fit
MIN_CELL_VAR = 0.02                 # 3x3 neighbourhood variance floor, in cell units squared
MIN_CELL_PIXELS = 6

VERIFY_PAD = 0.25                   # crop margin around the quad, fraction of height_px
VERIFY_TEXT_PX = 48                 # upscale crops so the text is about this tall
VERIFY_MAX_UP = 8.0
VERIFY_MAX_W = 2400


def write_json_atomic(path, text):
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def dumps_text_json(doc):
    """Header keys one per line, then one item per line (a full indent explodes the grids)."""
    head = {k: v for k, v in doc.items() if k != "items"}
    lines = ["{"]
    for k, v in head.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    items = [json.dumps(it) for it in doc.get("items", [])]
    lines.append('  "items": [' + ("" if items else "]"))
    for k, s in enumerate(items):
        lines.append("    " + s + ("," if k + 1 < len(items) else ""))
    if items:
        lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ────────────────────────────────────────────────────────────────────────────
# GLB baseColor texture (stdlib json + struct)
# ────────────────────────────────────────────────────────────────────────────

def read_glb(path):
    """GLB (or plain .gltf JSON) -> (gltf dict, BIN chunk bytes or None)."""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != GLB_MAGIC:
        return json.loads(data.decode("utf-8")), None
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise ValueError(f"{path}: GLB version {version}, expected 2")
    off, gltf, bin_chunk = 12, None, None
    end = min(length, len(data))
    while off + 8 <= end:
        clen, ctype = struct.unpack_from("<II", data, off)
        body = data[off + 8:off + 8 + clen]
        if ctype == CHUNK_JSON and gltf is None:
            gltf = json.loads(body.decode("utf-8"))
        elif ctype == CHUNK_BIN and bin_chunk is None:
            bin_chunk = body
        off += 8 + clen
    if gltf is None:
        raise ValueError(f"{path}: no JSON chunk")
    return gltf, bin_chunk


def _uri_bytes(uri, base_dir):
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return (pathlib.Path(base_dir) / unquote(uri)).read_bytes()


def image_bytes(gltf, bin_chunk, image_idx, base_dir="."):
    """Encoded bytes of gltf["images"][image_idx]: bufferView (GLB BIN or a buffer uri) or uri."""
    img = gltf["images"][image_idx]
    if "bufferView" in img:
        bv = gltf["bufferViews"][img["bufferView"]]
        buf = gltf["buffers"][bv["buffer"]]
        if buf.get("uri") is None:
            if bin_chunk is None:
                raise ValueError("image bufferView points at the GLB BIN chunk, but there is none")
            data = bin_chunk
        else:
            data = _uri_bytes(buf["uri"], base_dir)
        o = int(bv.get("byteOffset", 0))
        return data[o:o + int(bv["byteLength"])]
    if "uri" in img:
        return _uri_bytes(img["uri"], base_dir)
    raise ValueError(f"image {image_idx} has neither bufferView nor uri")


def _texture_source(tex):
    if "source" in tex:
        return int(tex["source"])
    for ext in ("EXT_texture_webp", "KHR_texture_basisu"):
        src = ((tex.get("extensions") or {}).get(ext) or {}).get("source")
        if src is not None:
            return int(src)
    raise ValueError(f"texture {tex} has no image source")


def base_color_texture(glb_path):
    """GLB / glTF -> (PIL image, info). The meshes must use exactly one baseColor image.

    Raises ValueError for layouts the uv lookup cannot follow: several baseColor images (their
    uv spaces overlap), a baseColor on TEXCOORD_n with n > 0 (mvgen writes the active UV layer)
    and KHR_texture_transform (Blender applies it in the shader, not in the UV data).
    """
    gltf, bin_chunk = read_glb(glb_path)
    mats = gltf.get("materials") or []
    used = sorted({p["material"] for m in gltf.get("meshes") or [] for p in m.get("primitives") or []
                   if p.get("material") is not None}) or list(range(len(mats)))
    refs = {}
    for mi in used:
        bct = (mats[mi].get("pbrMetallicRoughness") or {}).get("baseColorTexture")
        if bct is None:
            continue
        if int(bct.get("texCoord", 0)) != 0:
            raise ValueError(f"material {mi} samples baseColor from TEXCOORD_{bct['texCoord']}")
        if "KHR_texture_transform" in (bct.get("extensions") or {}):
            raise ValueError(f"material {mi} uses KHR_texture_transform")
        tex = gltf["textures"][int(bct["index"])]
        src = _texture_source(tex)
        samp = (gltf.get("samplers") or [])[tex["sampler"]] if "sampler" in tex else {}
        refs.setdefault(src, {"materials": [], "wrap": [int(samp.get("wrapS", WRAP_REPEAT)),
                                                        int(samp.get("wrapT", WRAP_REPEAT))]})
        refs[src]["materials"].append(mi)
    if not refs:
        raise ValueError(f"{glb_path}: no material has a baseColorTexture")
    if len(refs) > 1:
        raise ValueError(f"{glb_path}: materials use {len(refs)} different baseColor images")
    src, ref = next(iter(refs.items()))
    img = Image.open(io.BytesIO(image_bytes(gltf, bin_chunk, src, pathlib.Path(glb_path).parent)))
    img.load()
    info = {"glb": str(glb_path), "image": src, "materials": ref["materials"], "wrap": ref["wrap"],
            "mime": gltf["images"][src].get("mimeType"), "size": [img.width, img.height]}
    return img, info


def wrap_coord(c, mode):
    if mode == WRAP_CLAMP:
        return np.clip(c, 0.0, np.nextafter(1.0, 0.0))
    if mode == WRAP_MIRROR:
        f = np.mod(c, 2.0)
        return np.where(f > 1.0, 2.0 - f, f)
    return c - np.floor(c)


def uv_to_texel(uv, tex_wh, wrap=(WRAP_REPEAT, WRAP_REPEAT)):
    """mvgen uv (Blender, v up) -> continuous texture pixel xy, top-left origin.

    The glTF importer stores v_blender = 1 - v_gltf and glTF v runs top-down, so texture pixel =
    (u * W, (1 - v) * H) (mvgen metadata "encoding.uv"). Checked per sku against the albedo
    render (uv_check), since a wrong flip would put every box on the wrong text.
    """
    uv = np.asarray(uv, np.float64)
    W, H = tex_wh
    u = wrap_coord(uv[..., 0], wrap[0])
    v = wrap_coord(1.0 - uv[..., 1], wrap[1])
    return np.stack([u * W, v * H], axis=-1)


# ────────────────────────────────────────────────────────────────────────────
# CFI-3DGen template islands -> provenance
# ────────────────────────────────────────────────────────────────────────────

def load_template(path, tex_wh):
    """template_meta.json -> islands in texture px and the front region.

    Island bbox is [x, y, w, h] in canvas px, and for circle islands (x, y) is the centre
    (uv6_v2.py pack_can: atlas_x = caps_ox + cap_d // 2). Texture px = canvas px *
    (tex_w / canvas_w, tex_h / canvas_h), since Gemini's output only keeps the canvas aspect.
    """
    with open(path) as f:
        meta = json.load(f)
    W, H = tex_wh
    sx, sy = W / float(meta["canvas_w"]), H / float(meta["canvas_h"])
    islands, front = [], None
    for isl in meta.get("islands") or []:
        x, y, w, h = [float(v) for v in isl["bbox"]]
        shape = isl.get("shape", "rect")
        if shape == "circle":
            box = [(x - w / 2) * sx, (y - h / 2) * sy, (x + w / 2) * sx, (y + h / 2) * sy]
        else:
            box = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
        islands.append({"name": isl["name"], "shape": shape, "box": [round(v, 2) for v in box]})
        if isl["name"] == "front":
            front = list(box)
        elif isl["name"] == "body" and meta.get("layout") == "can":
            k, n = CAN_BODY_SECTIONS.index("FRONT"), len(CAN_BODY_SECTIONS)
            bw = box[2] - box[0]
            front = [box[0] + bw * k / n, box[1], box[0] + bw * (k + 1) / n, box[3]]
    return {"path": str(path), "layout": meta.get("layout"), "canvas": [meta["canvas_w"], meta["canvas_h"]],
            "scale": [round(sx, 5), round(sy, 5)],
            "aspect_mismatch": round(abs(sx / sy - 1.0), 4),
            "islands": islands, "front": None if front is None else [round(v, 2) for v in front]}


def _inside(pt, shape, box):
    x, y = pt
    if shape == "circle":
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rx, ry = max((box[2] - box[0]) / 2, 1e-9), max((box[3] - box[1]) / 2, 1e-9)
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def island_of(pt, template):
    for isl in template["islands"]:
        if _inside(pt, isl["shape"], isl["box"]):
            return isl["name"]
    return None


def provenance_of(pt, template):
    """Provenance of a texture point: "front" inside the front region, else "generated" (text on
    every other island, and outside all islands, is Gemini's invention), "unknown" without a
    template."""
    if template is None:
        return "unknown"
    fr = template["front"]
    return "front" if fr is not None and _inside(pt, "rect", fr) else "generated"


# ────────────────────────────────────────────────────────────────────────────
# Items: OCR lines in texture pixels
# ────────────────────────────────────────────────────────────────────────────

def reading_angle(quad):
    """Baseline direction in degrees, image frame (y down): 0 reads right, 90 down, -90 up."""
    q = np.asarray(quad, np.float64)
    d = (q[1] - q[0]) + (q[2] - q[3])
    return float(math.degrees(math.atan2(d[1], d[0])))


def item_flags(text, conf, angle_deg):
    flags = []
    if TEMPLATE_LEAK_RE.search((text or "").translate(HOMOGLYPHS)):
        flags.append("template_leak")
    if conf < LOW_CONF:
        flags.append("low_conf")
    if abs(angle_deg) > ROTATED_TOL_DEG:
        flags.append("rotated")
    return flags


def build_items(lines, template=None):
    """OCR lines (unitex.ocr items, texture px) -> text.json items without views."""
    items = []
    for ln in lines:
        text = " ".join(str(ln["text"]).split())
        if not any(c.isalnum() for c in text):
            continue
        q = np.asarray(ln["quad"], np.float64).reshape(4, 2)
        angle = reading_angle(q)
        centre = q.mean(0)
        items.append({
            "id": len(items),
            "text": text,
            "conf": round(float(ln["conf"]), 4),
            "provenance": provenance_of(centre, template),
            "flags": item_flags(text, float(ln["conf"]), angle),
            "angle_deg": round(angle, 2),
            "src_quad": [[round(float(x), 2), round(float(y), 2)] for x, y in q],
            "island": island_of(centre, template) if template is not None else None,
            "views": {},
        })
    return items


def quad_area(q):
    q = np.asarray(q, np.float64)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def instance_map(quads, tex_wh):
    """int32 (H, W) map, -1 = no item, k = quads[k]: texel k belongs to the quad holding its
    centre, tested with the same inverse bilinear map the (s, t) coordinates use. Larger quads
    are written first so a small item inside a big one keeps its texels."""
    W, H = int(tex_wh[0]), int(tex_wh[1])
    idmap = np.full((H, W), -1, np.int32)
    for k in sorted(range(len(quads)), key=lambda k: -quad_area(quads[k])):
        q = np.asarray(quads[k], np.float64)
        x0, x1 = max(int(np.floor(q[:, 0].min())), 0), min(int(np.ceil(q[:, 0].max())), W)
        y0, y1 = max(int(np.floor(q[:, 1].min())), 0), min(int(np.ceil(q[:, 1].max())), H)
        if x1 <= x0 or y1 <= y0 or quad_area(q) < 1e-6:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        st = inverse_bilinear(np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1), q)
        inside = ((st >= 0.0) & (st <= 1.0)).all(1).reshape(yy.shape)
        idmap[y0:y1, x0:x1][inside] = k
    return idmap


# ────────────────────────────────────────────────────────────────────────────
# Reading-frame coordinates
# ────────────────────────────────────────────────────────────────────────────

def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def bilerp(quad, s, t):
    """Reading-order quad (TL, TR, BR, BL) at (s, t): s along the baseline, t down the letters."""
    tl, tr, br, bl = [np.asarray(p, np.float64) for p in quad]
    s = np.asarray(s, np.float64)[..., None]
    t = np.asarray(t, np.float64)[..., None]
    return (1 - s) * (1 - t) * tl + s * (1 - t) * tr + s * t * br + (1 - s) * t * bl


def inverse_bilinear(p, quad):
    """Points p (n, 2) -> (s, t) (n, 2) with bilerp(quad, s, t) = p (closed form).

    p = tl + s e + t f + s t g with e = tr - tl, f = bl - tl, g = tl - tr + br - bl, so t solves
    cross(g, f) t^2 + (cross(e, f) + cross(h, g)) t + cross(h, e) = 0 (h = p - tl), and
    s = <h - t f, e + t g> / |e + t g|^2. The root nearer [0, 1] is kept.
    """
    tl, tr, br, bl = [np.asarray(q, np.float64) for q in quad]
    p = np.asarray(p, np.float64).reshape(-1, 2)
    e, f, g = tr - tl, bl - tl, tl - tr + br - bl
    h = p - tl
    k2 = float(_cross(g, f))
    k1 = float(_cross(e, f)) + _cross(h, g)
    k0 = _cross(h, e)
    scale = max(abs(float(_cross(e, f))), 1e-12)
    if abs(k2) < 1e-9 * scale:
        t = -k0 / np.where(np.abs(k1) > 1e-12, k1, 1e-12)
    else:
        sq = np.sqrt(np.maximum(k1 * k1 - 4 * k0 * k2, 0.0))
        t1, t2 = (-k1 - sq) / (2 * k2), (-k1 + sq) / (2 * k2)

        def dist(t_):
            return np.maximum(0.0, -t_) + np.maximum(0.0, t_ - 1.0)

        t = np.where(dist(t1) <= dist(t2), t1, t2)
    d = e[None] + t[:, None] * g[None]
    s = np.sum((h - t[:, None] * f[None]) * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-12)
    return np.stack([s, t], axis=1)


# ────────────────────────────────────────────────────────────────────────────
# Per-view geometry of one item
# ────────────────────────────────────────────────────────────────────────────

def _box3(a):
    """Sum over the 3 x 3 cell neighbourhood of (..., nt, ns) arrays, zero padded."""
    p = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)])
    nt, ns = a.shape[-2:]
    return sum(p[..., 1 + di:1 + di + nt, 1 + dj:1 + dj + ns] for di in (-1, 0, 1) for dj in (-1, 0, 1))


def _fit_from_moments(n, S):
    """Moments -> (mean_st (..., 2), mean_xy (..., 2), J (..., 2, 2), normalized cov (..., 2, 2)).

    S rows: s, t, x, y, ss, st, tt, sx, sy, tx, ty (sums). J = dxy/dst, columns (d/ds, d/dt).
    """
    nn = np.maximum(n, 1e-12)
    ms, mt, mx, my = S[0] / nn, S[1] / nn, S[2] / nn, S[3] / nn
    css, cst, ctt = S[4] / nn - ms * ms, S[5] / nn - ms * mt, S[6] / nn - mt * mt
    cxs, cys, cxt, cyt = S[7] / nn - ms * mx, S[8] / nn - ms * my, S[9] / nn - mt * mx, S[10] / nn - mt * my
    det = css * ctt - cst * cst
    idet = 1.0 / np.where(np.abs(det) > 1e-18, det, 1e-18)
    i00, i01, i11 = ctt * idet, -cst * idet, css * idet
    J = np.stack([np.stack([cxs * i00 + cxt * i01, cxs * i01 + cxt * i11], -1),
                  np.stack([cys * i00 + cyt * i01, cys * i01 + cyt * i11], -1)], -2)
    cov = np.stack([np.stack([css, cst], -1), np.stack([cst, ctt], -1)], -2)
    return np.stack([ms, mt], -1), np.stack([mx, my], -1), J, cov


def _moments(st, xy, index, size):
    s, t, x, y = st[:, 0], st[:, 1], xy[:, 0], xy[:, 1]
    comps = (s, t, x, y, s * s, s * t, t * t, s * x, s * y, t * x, t * y)
    n = np.bincount(index, minlength=size).astype(np.float64)
    S = np.stack([np.bincount(index, weights=c, minlength=size) for c in comps])
    return n, S


def _min_eig(cov):
    a, b, d = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
    return (a + d) / 2 - np.sqrt(((a - d) / 2) ** 2 + b * b)


def cell_model(st, xy, ns=NS, nt=NT):
    """Local affine maps of one item in one view, or None when the pixels are degenerate.

    Pixels are binned into the ns x nt (s, t) cells. Each hit cell keeps its mean (s, t), mean xy
    and a Jacobian J = dxy/dst fitted over its 3 x 3 cell neighbourhood (the item-wide J where
    that neighbourhood is too thin), so xy(p) ~ mean_xy + J (p - mean_st) near the cell.
    """
    n = len(st)
    if n < 3:
        return None
    n_all, S_all = _moments(st, xy, np.zeros(n, int), 1)
    _, _, J_g, cov_g = _fit_from_moments(n_all[0], S_all[:, 0])
    if _min_eig(cov_g) < MIN_ST_VAR or abs(np.linalg.det(J_g)) < 1e-9:
        return None
    ci = np.clip(np.floor(st[:, 0] * ns).astype(int), 0, ns - 1)
    ri = np.clip(np.floor(st[:, 1] * nt).astype(int), 0, nt - 1)
    cnt, S = _moments(st, xy, ri * ns + ci, ns * nt)
    cnt, S = cnt.reshape(nt, ns), S.reshape(-1, nt, ns)
    hit = cnt > 0
    mst, mxy, _, _ = _fit_from_moments(cnt, S)
    n3, S3 = _box3(cnt), _box3(S)
    _, _, J3, cov3 = _fit_from_moments(n3, S3)
    unit = np.array([ns, nt], np.float64)
    ok3 = (n3 >= MIN_CELL_PIXELS) & (_min_eig(cov3 * unit[:, None] * unit[None, :]) >= MIN_CELL_VAR)
    hr, hc = np.nonzero(hit)
    return {"ns": ns, "nt": nt, "unit": unit, "hit": hit, "hr": hr, "hc": hc,
            "mst": mst[hr, hc], "mxy": mxy[hr, hc],
            "J": np.where(ok3[..., None, None], J3, J_g)[hr, hc], "J_g": J_g}


def predict(model, pts):
    """(s, t) points (n, 2) -> view xy (n, 2) through the nearest hit cell's local map."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    d = ((((model["mst"][None] - pts[:, None]) * model["unit"]) ** 2).sum(-1))
    k = np.argmin(d, axis=1)
    return model["mxy"][k] + np.einsum("nij,nj->ni", model["J"][k], pts - model["mst"][k])


def _label_numpy(mask):
    """8-connected labels (1..n, 0 = background) by min-label propagation with pointer jumping."""
    H, W = mask.shape
    big = H * W
    lab = np.where(mask, np.arange(big).reshape(H, W), big)
    while True:
        p = np.pad(lab, 1, constant_values=big)
        m = lab
        for dy in range(3):
            for dx in range(3):
                m = np.minimum(m, p[dy:dy + H, dx:dx + W])
        flat = np.append(np.where(mask, m, big).ravel(), big)
        while True:
            j = flat[flat]
            if np.array_equal(j, flat):
                break
            flat = j
        m = flat[:-1].reshape(H, W)
        if np.array_equal(m, lab):
            break
        lab = m
    out = np.zeros((H, W), int)
    uniq, inv = np.unique(lab[mask], return_inverse=True)
    out[mask] = inv + 1
    return out, len(uniq)


def pixel_components(xy):
    """8-connected components of a view pixel set (pixel centres) -> per-pixel label, 0 = largest."""
    ix, iy = np.floor(xy[:, 0]).astype(int), np.floor(xy[:, 1]).astype(int)
    x0, y0 = ix.min(), iy.min()
    mask = np.zeros((iy.max() - y0 + 1, ix.max() - x0 + 1), bool)
    mask[iy - y0, ix - x0] = True
    try:
        from scipy.ndimage import label
        lab, _ = label(mask, structure=np.ones((3, 3), int))
    except ImportError:
        lab, _ = _label_numpy(mask)
    lp = lab[iy - y0, ix - x0]
    uniq, inv, counts = np.unique(lp, return_inverse=True, return_counts=True)
    rank = np.empty(len(uniq), int)
    rank[np.argsort(-counts, kind="stable")] = np.arange(len(uniq))
    return rank[inv]


def consistent_pixels(st, xy, ns=NS, nt=NT):
    """Keep mask: the largest connected pixel component plus every other component whose pixels
    sit where the largest one's local maps put their (s, t) (median error <= max(3 px, height/4)).

    The rest are the same texels rendered at another place (a can's rim faces unwrapped into
    the top of the body strip, reused UV islands), which would drag the fit off the text.
    Returns (keep, model of the kept pixels) or (None, None) when the main part is degenerate.
    """
    comp = pixel_components(xy)
    main = comp == 0
    model = cell_model(st[main], xy[main], ns, nt)
    if model is None or comp.max() == 0:
        return (np.ones(len(st), bool), model) if model is not None else (None, None)
    tol = max(3.0, 0.25 * float(np.median(np.linalg.norm(model["J"][:, :, 1], axis=-1))))
    keep = main.copy()
    for c in range(1, comp.max() + 1):
        sel = comp == c
        err = np.linalg.norm(predict(model, st[sel]) - xy[sel], axis=1)
        if float(np.median(err)) <= tol:
            keep |= sel
    if keep.sum() > main.sum():
        model = cell_model(st[keep], xy[keep], ns, nt)
    return keep, model


def measure_view(st, xy, cos, ns=NS, nt=NT, min_pixels=MIN_PIXELS, lookup_st=None):
    """One item in one view -> text.json view entry, or None (too few pixels / degenerate).

    st (n, 2) reading-frame coordinates in [0, 1], xy (n, 2) view pixel centres, cos (n,) |n_z|.
    lookup_st(xy (m, 2)) -> (m, 2): the item's (s, t) of the texel the view shows at xy (NaN off
    the mask), used to keep occluded hole cells null. Without it every bracketed hole is filled.
    """
    if len(st) < max(int(min_pixels), 3):
        return None
    keep, model = consistent_pixels(st, xy, ns, nt)
    if model is None or int(keep.sum()) < max(int(min_pixels), 3):
        return None
    dropped = int(len(st) - keep.sum())
    st, xy, cos = st[keep], xy[keep], cos[keep]
    n = len(st)
    unit, hit, hr, hc = model["unit"], model["hit"], model["hr"], model["hc"]

    # half a pixel's footprint in (s, t)
    Jinv = np.linalg.inv(model["J_g"])
    hs = 0.5 * (abs(Jinv[0, 0]) + abs(Jinv[0, 1]))
    ht = 0.5 * (abs(Jinv[1, 0]) + abs(Jinv[1, 1]))

    centres = np.stack(np.meshgrid((np.arange(ns) + 0.5) / ns, (np.arange(nt) + 0.5) / nt), -1)
    G = np.full((nt, ns, 2), np.nan)
    G[hr, hc] = model["mxy"] + np.einsum("nij,nj->ni", model["J"], centres[hr, hc] - model["mst"])
    # holes bracketed by hit cells along the row or the column (a small item inside this one
    # took the texels, or cells thinner than a pixel): filled from the nearest hit cell, but only
    # where the view really shows this item's texels there (an occluder hides the cell otherwise)
    holes = [(r, c) for r, c in zip(*np.nonzero(~hit))
             if (hit[r, :c].any() and hit[r, c + 1:].any()) or (hit[:r, c].any() and hit[r + 1:, c].any())]
    if holes:
        hr_, hc_ = np.array(holes).T
        pts = predict(model, centres[hr_, hc_])
        ok = np.ones(len(pts), bool)
        if lookup_st is not None:
            d = np.abs(lookup_st(pts) - centres[hr_, hc_])
            ok = (d[:, 0] <= max(1.0 / ns, 2 * hs)) & (d[:, 1] <= max(1.0 / nt, 2 * ht))
        G[hr_[ok], hc_[ok]] = pts[ok]

    # visible (s, t) range: pixel centres widened by half a pixel's footprint
    s_lo, s_hi = max(0.0, st[:, 0].min() - hs), min(1.0, st[:, 0].max() + hs)
    t_lo, t_hi = max(0.0, st[:, 1].min() - ht), min(1.0, st[:, 1].max() + ht)

    # coverage: s bins touched by any pixel's footprint
    lo = np.clip(np.floor((st[:, 0] - hs) * ns).astype(int), 0, ns - 1)
    hi = np.clip(np.floor((st[:, 0] + hs) * ns).astype(int), 0, ns - 1)
    diff = np.zeros(ns + 1, int)
    np.add.at(diff, lo, 1)
    np.add.at(diff, hi + 1, -1)
    coverage = float((np.cumsum(diff)[:ns] > 0).mean())

    quad = predict(model, [(s_lo, t_lo), (s_hi, t_lo), (s_hi, t_hi), (s_lo, t_hi)])
    heights = np.linalg.norm(model["J"][:, :, 1], axis=-1)
    if float(np.median(heights)) < MIN_HEIGHT_PX:
        return None                 # edge-on (grazing) sliver, no text can show there

    return {
        "bbox": [round(float(xy[:, 0].min() - 0.5), 2), round(float(xy[:, 1].min() - 0.5), 2),
                 round(float(xy[:, 0].max() + 0.5), 2), round(float(xy[:, 1].max() + 0.5), 2)],
        "quad": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in quad],
        "pixels": int(n),
        "coverage": round(coverage, 4),
        "cos": round(float(np.mean(cos)), 4),
        "height_px": round(float(np.median(heights)), 2),
        "st_range": [round(s_lo, 4), round(s_hi, 4), round(t_lo, 4), round(t_hi, 4)],
        "dropped_px": dropped,
        "grid": {"ns": int(ns), "nt": int(nt),
                 "xy": [[None if not np.isfinite(G[r, c, 0]) else
                         [round(float(G[r, c, 0]), 2), round(float(G[r, c, 1]), 2)]
                         for c in range(ns)] for r in range(nt)]},
    }


# ────────────────────────────────────────────────────────────────────────────
# Views
# ────────────────────────────────────────────────────────────────────────────

def normal_cos(normal_rgb):
    """000i_normal.png (RGB = trunc(255 (n_cam + 1) / 2)) -> |n_cam.z|, the cosine to the view."""
    n = (np.asarray(normal_rgb, np.float64)[..., :3] + 0.5) / 255.0 * 2.0 - 1.0
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    return np.abs(n[..., 2])


def load_view(render_dir, i):
    """(uv (H, W, 2) float64, mask (H, W) bool, cos (H, W), albedo (H, W, 3) uint8 or None)."""
    render_dir = pathlib.Path(render_dir)
    with np.load(render_dir / f"{i:04d}_uv.npz") as z:
        uv, mask = z["uv"].astype(np.float64), z["mask"].astype(bool)
    cos = normal_cos(np.asarray(Image.open(render_dir / f"{i:04d}_normal.png").convert("RGB")))
    ap = render_dir / f"{i:04d}_albedo.png"
    albedo = np.asarray(Image.open(ap).convert("RGB")) if ap.exists() else None
    return uv, mask, cos, albedo


def project_items(items, idmap, tex_wh, wrap, views, min_pixels=MIN_PIXELS, ns=NS, nt=NT):
    """Fill item["views"] from raw views [(uv, mask, cos, albedo)]. Returns per-view counts."""
    Ht, Wt = idmap.shape
    quads = [np.asarray(it["src_quad"], np.float64) for it in items]
    counts = {}

    def st_lookup(uv, mask, quad):
        def f(pts):
            ix, iy = np.floor(pts[:, 0]).astype(int), np.floor(pts[:, 1]).astype(int)
            on = (ix >= 0) & (iy >= 0) & (ix < mask.shape[1]) & (iy < mask.shape[0])
            out = np.full((len(pts), 2), np.nan)
            on[on] = mask[iy[on], ix[on]]
            if on.any():
                out[on] = inverse_bilinear(uv_to_texel(uv[iy[on], ix[on]], tex_wh, wrap), quad)
            return out
        return f

    for i, (uv, mask, cos, _) in enumerate(views):
        rows, cols = np.nonzero(mask)
        txy = uv_to_texel(uv[rows, cols], tex_wh, wrap)
        tx = np.clip(np.floor(txy[:, 0]).astype(int), 0, Wt - 1)
        ty = np.clip(np.floor(txy[:, 1]).astype(int), 0, Ht - 1)
        ids = idmap[ty, tx]
        sel = ids >= 0
        order = np.argsort(ids[sel], kind="stable")
        pix = np.nonzero(sel)[0][order]
        uniq, start = np.unique(ids[pix], return_index=True)
        bounds = list(start) + [len(pix)]
        n_view = 0
        for k, a, b in zip(uniq, bounds[:-1], bounds[1:]):
            p = pix[a:b]
            st = np.clip(inverse_bilinear(txy[p], quads[k]), 0.0, 1.0)
            xy = np.stack([cols[p] + 0.5, rows[p] + 0.5], axis=1).astype(np.float64)
            entry = measure_view(st, xy, cos[rows[p], cols[p]], ns, nt, min_pixels,
                                 lookup_st=st_lookup(uv, mask, quads[k]))
            if entry is not None:
                items[k]["views"][str(i)] = entry
                n_view += 1
        counts[str(i)] = n_view
    return counts


def uv_check(tex_rgb, views, wrap, max_pixels=40000):
    """Median |albedo - texture(uv)| per pixel (mean over channels, /255) with the documented v
    flip and without it. A correct convention gives a few levels, the wrong one tens."""
    Ht, Wt = tex_rgb.shape[:2]
    rng = np.random.default_rng(0)
    d_ok, d_flip = [], []
    for uv, mask, _, albedo in views:
        if albedo is None:
            continue
        rows, cols = np.nonzero(mask)
        if len(rows) > max_pixels:
            keep = rng.choice(len(rows), max_pixels, replace=False)
            rows, cols = rows[keep], cols[keep]
        a = albedo[rows, cols].astype(np.float64)
        for flip, acc in ((False, d_ok), (True, d_flip)):
            u = uv[rows, cols].copy()
            if flip:
                u[:, 1] = 1.0 - u[:, 1]
            txy = uv_to_texel(u, (Wt, Ht), wrap)
            tx = np.clip(np.floor(txy[:, 0]).astype(int), 0, Wt - 1)
            ty = np.clip(np.floor(txy[:, 1]).astype(int), 0, Ht - 1)
            acc.append(np.abs(tex_rgb[ty, tx].astype(np.float64) - a).mean(1))
    if not d_ok:
        return None
    return {"mad": round(float(np.median(np.concatenate(d_ok))), 3),
            "mad_vflip": round(float(np.median(np.concatenate(d_flip))), 3)}


# ────────────────────────────────────────────────────────────────────────────
# --verify-view-ocr
# ────────────────────────────────────────────────────────────────────────────

def _quad_len_height(q):
    q = np.asarray(q, np.float64)
    return (float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2,
            float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def verify_crop(albedo, view):
    """Rectified, upscaled crop of a view quad with a VERIFY_PAD margin.
    Returns (PIL RGB, (y0, y1) of the unpadded text band in crop pixels)."""
    q = np.asarray(view["quad"], np.float64)
    length, height = _quad_len_height(q)
    height = max(height, float(view.get("height_px") or 0.0), 1.0)
    length = max(length, 1.0)
    pad = VERIFY_PAD * height
    ps, pt = pad / length, pad / height
    big = [bilerp(q, s, t) for s, t in ((-ps, -pt), (1 + ps, -pt), (1 + ps, 1 + pt), (-ps, 1 + pt))]
    f = min(max(VERIFY_TEXT_PX / height, 1.0), VERIFY_MAX_UP)
    f = min(f, VERIFY_MAX_W / (length + 2 * pad))
    out_wh = (max(8, int(round((length + 2 * pad) * f))), max(8, int(round((height + 2 * pad) * f))))
    crop = crop_gt(albedo, big, out_wh)
    return crop, (pad * f, (pad + height) * f)


def verify_items(items, albedos, backend, backend_kwargs=None, key_prefix="", cache_dir=None):
    """OCR every item x view crop in one backend call and set view_ocr_text / view_ocr_ned."""
    jobs, crops, keys = [], [], []
    for it in items:
        for k, v in it["views"].items():
            alb = albedos[int(k)]
            if alb is None:
                continue
            crop, band = verify_crop(alb, v)
            jobs.append((it, k, band))
            crops.append(crop)
            keys.append(f"{key_prefix}{it['id']}_v{k}")
    if not crops:
        return 0
    results = ocr_mod.ocr_images(crops, backend, rotations=(0,), keys=keys, cache_dir=cache_dir,
                                 backend_kwargs=backend_kwargs)
    for (it, k, (y0, y1)), found in zip(jobs, results):
        # lines centred outside the text band are neighbours that the margin let in
        keep = [ln for ln in found if y0 <= (ln["bbox"][1] + ln["bbox"][3]) / 2 <= y1]
        text = " ".join(" ".join(ln["text"].split()) for ln in keep)
        it["views"][k]["view_ocr_text"] = text
        it["views"][k]["view_ocr_ned"] = round(ned(normalize(it["text"]), normalize(text)), 4)
    return len(crops)


# ────────────────────────────────────────────────────────────────────────────
# One sku
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RegionConfig:
    backend: object = "vision"               # name for unitex.ocr, or a backend object
    backend_kwargs: dict = field(default_factory=dict)
    rotations: tuple = (0, 90, 270)
    upscale_to: int = None
    min_conf: float = 0.0
    min_pixels: int = MIN_PIXELS
    verify: bool = False
    bundle_v2: str = None
    glbs: str = None
    glb_name: str = "textured.glb"
    cache_dir: str = None
    ns: int = NS
    nt: int = NT


def resolve_glb(meta, sku, cfg):
    src = meta.get("source_glb")
    if src and os.path.exists(src):
        return pathlib.Path(src)
    if cfg.glbs:
        p = pathlib.Path(cfg.glbs) / sku / cfg.glb_name
        if p.exists():
            return p
    raise FileNotFoundError(f"source GLB {src!r} not found (pass --glbs DIR for <DIR>/<sku>/{cfg.glb_name})")


def usable(view, cfg=GlyphConfig()):
    """glyph.py's per-view filter (GlyphConfig min_height_px, min_coverage, min_cos, defaults):
    the views that would become glyph instances. Only used for the stats."""
    return (float(view.get("height_px") or 0.0) >= cfg.min_height_px
            and float(view.get("coverage", 1.0)) >= cfg.min_coverage
            and float(view.get("cos", 1.0)) >= cfg.min_cos)


def _median(xs):
    return None if not xs else round(float(np.median(xs)), 4)


def summarize(items, view_counts):
    prov, leaks = {}, []
    for it in items:
        prov[it["provenance"]] = prov.get(it["provenance"], 0) + 1
        if "template_leak" in it["flags"]:
            leaks.append(it["text"])
    views = [(it, v) for it in items for v in it["views"].values()]
    front = [v for it, v in views if it["provenance"] == "front"]

    def neds(vs):
        return [v["view_ocr_ned"] for v in vs if "view_ocr_ned" in v]

    return {
        "items": len(items),
        "items_with_views": sum(1 for it in items if it["views"]),
        "items_per_view": view_counts,
        "item_views": len(views),
        "item_views_usable": sum(usable(v) for _, v in views),
        "provenance": prov,
        "template_leak": leaks,
        "flag_counts": {f: sum(f in it["flags"] for it in items) for f in ("template_leak", "low_conf", "rotated")},
        "median_view_ocr_ned_front": _median(neds(front)),
        "median_view_ocr_ned_front_usable": _median(neds([v for v in front if usable(v)])),
        "median_view_ocr_ned_all": _median(neds([v for _, v in views])),
        "n_verified": len(neds([v for _, v in views])),
    }


def process_sku(render_dir, cfg):
    """render/<volume>/<sku>/ -> text.json document (not written)."""
    render_dir = pathlib.Path(render_dir)
    with open(render_dir / "metadata.json") as f:
        meta = json.load(f)
    sku = str(meta.get("sku") or render_dir.name)
    glb = resolve_glb(meta, sku, cfg)
    tex, tex_info = base_color_texture(glb)
    tex_rgb = np.asarray(tex.convert("RGB"))
    tex_wh = (tex_rgb.shape[1], tex_rgb.shape[0])

    template = None
    if cfg.bundle_v2:
        tp = pathlib.Path(cfg.bundle_v2) / sku / "template" / "template_meta.json"
        if tp.exists():
            template = load_template(tp, tex_wh)
            if template["aspect_mismatch"] > 0.03:
                print(f"  [{sku}] warning: texture aspect differs from the template canvas by "
                      f"{100 * template['aspect_mismatch']:.1f}%")
        else:
            print(f"  [{sku}] no {tp}, provenance stays unknown")

    lines = ocr_mod.ocr_images([Image.fromarray(tex_rgb)], cfg.backend, rotations=cfg.rotations,
                               upscale_to=cfg.upscale_to, min_conf=cfg.min_conf, keys=[sku],
                               cache_dir=cfg.cache_dir, backend_kwargs=cfg.backend_kwargs)[0]
    items = build_items(lines, template)

    views = [load_view(render_dir, i) for i in range(6)]
    check = uv_check(tex_rgb, views, tex_info["wrap"])
    if check and check["mad"] > 12 and check["mad_vflip"] < 0.5 * check["mad"]:
        raise RuntimeError(f"uv convention check failed: albedo vs texture(uv) {check['mad']}, "
                           f"v-flipped {check['mad_vflip']}")
    idmap = instance_map([it["src_quad"] for it in items], tex_wh)
    counts = project_items(items, idmap, tex_wh, tex_info["wrap"], views, cfg.min_pixels, cfg.ns, cfg.nt)
    if cfg.verify:
        verify_items(items, [v[3] for v in views], cfg.backend, cfg.backend_kwargs,
                     key_prefix=f"{sku}_", cache_dir=cfg.cache_dir)

    be = cfg.backend
    return {
        "version": TEXT_JSON_VERSION,
        "res": int(meta.get("res") or views[0][1].shape[0]),
        "source": "uv-ocr",
        "sku": sku,
        "texture": {k: tex_info[k] for k in ("glb", "image", "size", "wrap", "mime")},
        "ocr": {"backend": be if isinstance(be, str) else getattr(be, "name", str(be)),
                "rotations": list(cfg.rotations), "upscale_to": cfg.upscale_to, "min_conf": cfg.min_conf,
                "lines": len(lines), "verify_view_ocr": bool(cfg.verify)},
        "template": None if template is None else {k: template[k] for k in
                                                   ("path", "layout", "canvas", "scale", "front", "islands")},
        "uv_check": check,
        "grid_note": "xy[it][is], t = 0 at the top of the letters, s = 0 at the first letter",
        "stats": summarize(items, counts),
        "items": items,
    }


# ────────────────────────────────────────────────────────────────────────────
# Debug drawing
# ────────────────────────────────────────────────────────────────────────────

def _colour(k):
    r, g, b = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_debug(render_dir, doc, legend_rows=150):
    """Albedo strip (FULL_INDEX order) with every item's per-view quad (thick edge = baseline
    top, dot = TL corner) and grid cell centres, plus a legend. Returns a PIL image."""
    render_dir = pathlib.Path(render_dir)
    views = [np.asarray(Image.open(render_dir / f"{i:04d}_albedo.png").convert("RGB")) for i in range(6)]
    res = views[0].shape[0]
    strip = Image.fromarray(stack_strip(views)).convert("RGBA")
    over = Image.new("RGBA", strip.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for s in range(6):
        d.line([(s * res, 0), (s * res, res)], fill=(255, 200, 0, 255), width=2)
        d.text((s * res + 4, 4), f"slot {s} {STRIP_NAMES[s]}", fill=(0, 0, 0, 255))
    for it in doc["items"]:
        col = _colour(it["id"])
        for k, v in it["views"].items():
            off = raw_to_slot(int(k)) * res
            q = [(x + off, y) for x, y in v["quad"]]
            d.line(q + [q[0]], fill=col + (255,), width=1)
            d.line([q[0], q[1]], fill=col + (255,), width=3)
            d.ellipse((q[0][0] - 2, q[0][1] - 2, q[0][0] + 2, q[0][1] + 2), fill=col + (255,))
            for row in v["grid"]["xy"]:
                for p in row:
                    if p is not None:
                        d.point((p[0] + off, p[1]), fill=(0, 0, 0, 255))
            d.text((q[0][0] + 2, q[0][1] - 11), f"#{it['id']}", fill=col + (255,))
    top = Image.alpha_composite(strip, over).convert("RGB")

    rows = [it for it in doc["items"] if it["views"]][:legend_rows]
    lh = 13
    out = Image.new("RGB", (top.width, res + lh * (len(rows) + 1) + 8), (25, 25, 25))
    out.paste(top, (0, 0))
    dd = ImageDraw.Draw(out)
    for n, it in enumerate(rows):
        vs = " ".join(f"v{k}:{v['coverage']:.2f}/{v['height_px']:.0f}px"
                      + (f"/ned{v['view_ocr_ned']:.2f}" if "view_ocr_ned" in v else "")
                      for k, v in it["views"].items())
        dd.text((6, res + 6 + n * lh),
                f"#{it['id']:3d} [{it['provenance'][:3]}] {','.join(it['flags']) or '-'}  {it['text'][:60]!r}  {vs}",
                fill=_colour(it["id"]))
    return out


def draw_texture_debug(glb, doc, max_side=2048):
    """Texture with src_quads (thick edge = baseline top) and template islands / front region."""
    tex, _ = base_color_texture(glb)
    tex = tex.convert("RGB")
    f = min(1.0, max_side / max(tex.size))
    img = tex.resize((round(tex.width * f), round(tex.height * f))) if f < 1 else tex.copy()
    d = ImageDraw.Draw(img)
    tpl = doc.get("template")
    if tpl:
        for isl in tpl["islands"]:
            b = [v * f for v in isl["box"]]
            (d.ellipse if isl["shape"] == "circle" else d.rectangle)(b, outline=(255, 255, 0), width=2)
        if tpl.get("front"):
            d.rectangle([v * f for v in tpl["front"]], outline=(0, 255, 0), width=4)
    for it in doc["items"]:
        col = _colour(it["id"])
        q = [(x * f, y * f) for x, y in it["src_quad"]]
        d.line(q + [q[0]], fill=col, width=2)
        d.line([q[0], q[1]], fill=col, width=4)
        d.text((q[0][0], q[0][1] - 11), f"#{it['id']}", fill=col)
    return img


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def read_sku_list(spec):
    """Comma list or a file with one sku per line ("#" starts a comment)."""
    if spec and os.path.exists(spec):
        with open(spec) as f:
            return [ln.split("#", 1)[0].split()[0] for ln in f if ln.split("#", 1)[0].strip()]
    return [s.strip() for s in str(spec).split(",") if s.strip()]


def log_error(root, key, text):
    with open(pathlib.Path(root) / "text_regions_errors.log", "a") as f:
        f.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {key}\n{text}\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="mvgen train output root (render/<volume>/<sku>/)")
    p.add_argument("--volume", default="cfi")
    p.add_argument("--skus", default=None, help="comma list or file (default: every rendered sku)")
    p.add_argument("--bundle-v2", default=None,
                   help="CFI-3DGen approved_bundle_v2 dir (<sku>/template/template_meta.json) for provenance")
    ocr_mod.add_ocr_args(p)
    p.add_argument("--rotations", default="0,90,270", help="texture OCR rotations, CCW degrees")
    p.add_argument("--upscale-to", type=int, default=None, help="upscale the texture before OCR")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="drop OCR lines below this (lines under 0.5 are kept and flagged low_conf)")
    p.add_argument("--verify-view-ocr", action="store_true",
                   help="OCR each view quad crop of 000i_albedo.png (view_ocr_text, view_ocr_ned)")
    p.add_argument("--min-pixels", type=int, default=MIN_PIXELS, help="fewest view pixels to emit a view")
    p.add_argument("--glbs", default=None, help="fallback when metadata source_glb is missing: <glbs>/<sku>/<glb-name>")
    p.add_argument("--glb-name", default="textured.glb")
    p.add_argument("--ocr-cache", default=None, help="cache OCR results here (keyed by image and options)")
    p.add_argument("--debug-dir", default=None, help="write <sku>_views.png and <sku>_texture.png here")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    root = pathlib.Path(args.root)
    base = root / "render" / args.volume
    if args.skus:
        skus = read_sku_list(args.skus)
    else:
        skus = sorted(m.parent.name for m in base.glob("*/metadata.json"))
    if args.limit:
        skus = skus[:args.limit]
    cfg = RegionConfig(backend=args.backend, backend_kwargs=ocr_mod.backend_kwargs_from_args(args),
                       rotations=ocr_mod.parse_rotations(args.rotations), upscale_to=args.upscale_to,
                       min_conf=args.min_conf, min_pixels=args.min_pixels, verify=args.verify_view_ocr,
                       bundle_v2=args.bundle_v2, glbs=args.glbs, glb_name=args.glb_name,
                       cache_dir=args.ocr_cache)
    print(f"text_regions: {len(skus)} skus, backend {args.backend}, rotations {list(cfg.rotations)}, "
          f"verify {cfg.verify}, bundle-v2 {args.bundle_v2}")
    n_ok = n_skip = n_fail = 0
    t_run = time.time()
    for sku in skus:
        d = base / sku
        out = d / "text.json"
        if not (d / "metadata.json").exists():
            print(f"  [{sku}] no metadata.json (not rendered or incomplete), skipped")
            n_skip += 1
            continue
        if out.exists() and not args.overwrite:
            print(f"  [{sku}] text.json exists, skipped (--overwrite to redo)")
            n_skip += 1
            continue
        try:
            t = time.time()
            doc = process_sku(d, cfg)
            write_json_atomic(out, dumps_text_json(doc))
            st = doc["stats"]
            print(f"  [{sku}] {st['items']} items ({st['items_with_views']} visible), per view "
                  f"{st['items_per_view']}, provenance {st['provenance']}, leaks {st['template_leak']}, "
                  f"front ned {st['median_view_ocr_ned_front']} (usable views "
                  f"{st['median_view_ocr_ned_front_usable']}), uv_check {doc['uv_check']}, "
                  f"{time.time() - t:.1f} s")
            if args.debug_dir:
                dd = pathlib.Path(args.debug_dir)
                dd.mkdir(parents=True, exist_ok=True)
                draw_debug(d, doc).save(dd / f"{sku}_views.png")
                draw_texture_debug(doc["texture"]["glb"], doc).save(dd / f"{sku}_texture.png")
            n_ok += 1
        except Exception:
            tb = traceback.format_exc()
            print(f"  [{sku}] FAILED, traceback in text_regions_errors.log\n{tb}")
            log_error(root, sku, tb)
            n_fail += 1
    print(f"text_regions: {n_ok} written, {n_skip} skipped, {n_fail} failed in {time.time() - t_run:.1f} s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
