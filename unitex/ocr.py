"""
Backend-agnostic OCR for the UniTEX text-fidelity evaluation.

Every backend returns the same items, sorted in reading order:
  {"text": str, "conf": float in [0, 1],
   "quad": [[x, y] x 4] pixel coords of the input image, top-left origin, ordered
           top-left, top-right, bottom-right, bottom-left in the text's reading frame,
   "bbox": [x0, y0, x1, y1], "height": reading-frame line height px, "rot": rotation used}

Backends:
  paddle  PaddleOCR (lazy import). Supports 2.x ocr() and 3.x predict() output, textline
          angle classification on. What GlyphAnchor's InfoTextBench reports, use it for metrics.
  vision  macOS Apple Vision via unitex/vision_ocr.swift, compiled with /usr/bin/swiftc on
          first use into $UNITEX_CACHE_DIR (default ~/.cache/unitex). Vision returns normalized
          quads with a BOTTOM-left origin, converted here. Local triage only.
  json    precomputed results: <dir>/<key>.json with key = explicit key or the image stem.
          Accepts this module's output, raw vision_ocr output and the data-audit
          {"w","h","items":[{"t","c","bb"}]} format.

Options: upscale small inputs before OCR (upscale_to = target longest side, never downscales),
rotations (0 / 90 / 180 / 270 counter-clockwise, merged with NMS over quads so vertical side
panel text is found), min_conf.

CLI:
  python -m unitex.ocr --backend vision img.png
  python -m unitex.ocr --backend paddle --rotations 0,90,270 --upscale-to 2048 a.png b.png
"""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import tempfile

import numpy as np
from PIL import Image

try:
    from unitex.text_metrics import ned, normalize
except ImportError:          # run as a plain script from inside unitex/
    from text_metrics import ned, normalize

BACKENDS = ("paddle", "vision", "json")
SWIFT_SRC = pathlib.Path(__file__).with_name("vision_ocr.swift")


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

def quad_bbox(q):
    q = np.asarray(q, dtype=np.float64)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def quad_height(q):
    """Line height in the reading frame: mean of the two side edges (TL-BL, TR-BR)."""
    q = np.asarray(q, dtype=np.float64)
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def quad_length(q):
    """Baseline length in the reading frame: mean of the top and bottom edges."""
    q = np.asarray(q, dtype=np.float64)
    return float((np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_containment(a, b):
    """Intersection over the smaller box area."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / m if m > 0 else 0.0


def unrotate_points(xy, rot, w, h):
    """Points in an image rotated by `rot` degrees counter-clockwise (np.rot90(img, rot // 90))
    back to the original image of size (w, h). Continuous pixel coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    k = (rot // 90) % 4
    if k == 0:
        ox, oy = x, y
    elif k == 1:        # original (x, y) -> rotated (y, w - x)
        ox, oy = w - y, x
    elif k == 2:        # original (x, y) -> rotated (w - x, h - y)
        ox, oy = w - x, h - y
    else:               # original (x, y) -> rotated (h - y, x)
        ox, oy = y, h - x
    return np.stack([ox, oy], axis=-1)


def vision_quad_to_pixels(quad_norm, w, h):
    """Vision normalized quad (origin bottom-left, y up) -> top-left-origin pixel quad."""
    q = np.asarray(quad_norm, dtype=np.float64)
    return np.stack([q[:, 0] * w, (1.0 - q[:, 1]) * h], axis=-1)


def make_item(text, conf, quad, rot=0):
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return {
        "text": str(text),
        "conf": float(conf),
        "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in q],
        "bbox": [round(v, 2) for v in quad_bbox(q)],
        "height": round(quad_height(q), 2),
        "rot": int(rot),
    }


# ────────────────────────────────────────────────────────────────────────────
# Image input
# ────────────────────────────────────────────────────────────────────────────

def load_rgb(image, bg=(255, 255, 255)):
    """Path, PIL image or array -> RGB PIL image. Alpha is composited over `bg`."""
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
        image.load()
    elif isinstance(image, np.ndarray):
        a = image
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        image = Image.fromarray(a)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, tuple(bg) + (255,))
        return Image.alpha_composite(base, rgba).convert("RGB")
    return image.convert("RGB")


def upscale_factor(size, upscale_to, max_upscale=4.0):
    if not upscale_to:
        return 1.0
    f = min(float(max_upscale), float(upscale_to) / max(size))
    return f if f > 1.0 else 1.0


# ────────────────────────────────────────────────────────────────────────────
# Backends
# ────────────────────────────────────────────────────────────────────────────

class VisionBackend:
    """Apple Vision (macOS). recognize_batch runs one vision_ocr process for all images."""

    name = "vision"

    def __init__(self, langs=None, fast=False, correction=False, min_height=None, binary=None):
        if platform.system() != "Darwin":
            raise RuntimeError("the vision OCR backend needs macOS (Apple Vision)")
        self.langs = langs
        self.fast = fast
        self.correction = correction
        self.min_height = min_height
        self.binary = binary or os.environ.get("UNITEX_VISION_BIN") or str(self._build())

    @staticmethod
    def _build():
        src = SWIFT_SRC.read_bytes()
        digest = hashlib.sha1(src).hexdigest()[:10]
        cache = pathlib.Path(os.environ.get("UNITEX_CACHE_DIR", pathlib.Path.home() / ".cache" / "unitex"))
        exe = cache / f"vision_ocr_{digest}"
        if exe.exists():
            return exe
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"vision_ocr_{digest}.{os.getpid()}.tmp"
        print(f"  [ocr] compiling {SWIFT_SRC.name} -> {exe}")
        r = subprocess.run(["/usr/bin/swiftc", "-O", str(SWIFT_SRC), "-o", str(tmp)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"swiftc failed:\n{r.stderr}")
        os.replace(tmp, exe)
        return exe

    def signature(self):
        return {"backend": self.name, "langs": self.langs, "fast": self.fast,
                "correction": self.correction, "min_height": self.min_height}

    def recognize_batch(self, images):
        cmd = [self.binary]
        if self.langs:
            cmd += ["--langs", ",".join(self.langs)]
        if self.fast:
            cmd.append("--fast")
        if self.correction:
            cmd.append("--correction")
        if self.min_height is not None:
            cmd += ["--min-height", str(self.min_height)]
        with tempfile.TemporaryDirectory(prefix="unitex_ocr_") as td:
            paths = []
            for i, im in enumerate(images):
                p = os.path.join(td, f"{i:03d}.png")
                im.save(p)
                paths.append(p)
            r = subprocess.run(cmd + paths, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"vision_ocr failed ({r.returncode}): {r.stderr.strip()}")
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
        if len(rows) != len(images):
            raise RuntimeError(f"vision_ocr returned {len(rows)} results for {len(images)} images")
        out = []
        for row in rows:
            if "error" in row:
                raise RuntimeError(f"vision_ocr: {row['error']} ({row.get('path')})")
            w, h = row["w"], row["h"]
            out.append([(it["text"], it["conf"], vision_quad_to_pixels(it["quad_norm"], w, h))
                        for it in row["items"]])
        return out


def _paddle_quad(poly):
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) == 4:
        return p
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def parse_paddle2(result):
    """PaddleOCR 2.x ocr() output -> [(text, conf, quad)].

    2.6+ returns [page] with page = [[quad, (text, conf)], ...] or None, older versions
    return the page directly.
    """
    if not result:
        return []
    pages = result
    first = result[0]
    if (isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], (list, tuple))
            and len(first[1]) == 2 and isinstance(first[1][0], str)):
        pages = [result]
    out = []
    for page in pages:
        if not page:
            continue
        for line in page:
            quad, (text, conf) = line[0], line[1]
            out.append((text, float(conf), _paddle_quad(quad)))
    return out


def _res_get(res, key):
    try:
        v = res[key]
        if v is not None:
            return v
    except Exception:
        pass
    j = getattr(res, "json", None)
    if callable(j):
        j = j()
    if isinstance(j, dict):
        j = j.get("res", j)
        return j.get(key)
    return None


def parse_paddle3(results):
    """PaddleOCR 3.x predict() output (list of OCRResult dicts) -> [(text, conf, quad)]."""
    out = []
    for res in results or []:
        texts = _res_get(res, "rec_texts") or []
        scores = _res_get(res, "rec_scores")
        polys = _res_get(res, "rec_polys")
        if polys is None or len(polys) == 0:
            polys = _res_get(res, "dt_polys")
        boxes = _res_get(res, "rec_boxes")
        for i, t in enumerate(texts):
            conf = float(scores[i]) if scores is not None and i < len(scores) else 1.0
            if polys is not None and i < len(polys):
                q = _paddle_quad(polys[i])
            elif boxes is not None and i < len(boxes):
                x0, y0, x1, y1 = [float(v) for v in boxes[i]]
                q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            else:
                continue
            out.append((t, conf, q))
    return out


class PaddleBackend:
    """PaddleOCR 2.x or 3.x. Images are passed as BGR arrays, as both versions expect."""

    name = "paddle"

    def __init__(self, lang="en", det_limit_side_len=None, device=None, **extra):
        import paddleocr
        from paddleocr import PaddleOCR
        ver = str(getattr(paddleocr, "__version__", "2"))
        self.version = ver
        self.major = int(ver.split(".")[0]) if ver.split(".")[0].isdigit() else 2
        self.lang = lang
        self.det_limit_side_len = det_limit_side_len
        if self.major >= 3:
            kw = dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=True)
            if det_limit_side_len:
                kw.update(text_det_limit_side_len=int(det_limit_side_len), text_det_limit_type="max")
            if device:
                kw["device"] = device
        else:
            # 2.x resizes the longest side to 960 by default, which drops photo fine print
            kw = dict(lang=lang, use_angle_cls=True, show_log=False,
                      det_limit_side_len=int(det_limit_side_len or 2048), det_limit_type="max")
            if device:
                kw["use_gpu"] = device.startswith("gpu") or device.startswith("cuda")
        kw.update(extra)
        self.engine = PaddleOCR(**kw)

    def signature(self):
        return {"backend": self.name, "version": self.version, "lang": self.lang,
                "det_limit_side_len": self.det_limit_side_len}

    def recognize_batch(self, images):
        out = []
        for im in images:
            bgr = np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])
            if self.major >= 3:
                out.append(parse_paddle3(self.engine.predict(bgr)))
            else:
                out.append(parse_paddle2(self.engine.ocr(bgr, cls=True)))
        return out


def parse_json_result(d, size=None):
    """Any supported precomputed format -> [(text, conf, quad_px)], rescaled to `size` (w, h)."""
    if isinstance(d, list):
        items, w, h = d, None, None
    else:
        items = d.get("lines") or d.get("items") or []
        w, h = d.get("w") or (d.get("size") or [None, None])[0], d.get("h") or (d.get("size") or [None, None])[1]
    out = []
    for it in items:
        text = it.get("text", it.get("t", ""))
        conf = it.get("conf", it.get("c", 1.0))
        if "quad" in it:
            q = np.asarray(it["quad"], dtype=np.float64)
        elif "quad_norm" in it:
            q = vision_quad_to_pixels(it["quad_norm"], w, h)
        elif "bb" in it:           # data-audit format: normalized [minX, minY, w, h], bottom-left origin
            x, y, bw, bh = it["bb"]
            q = vision_quad_to_pixels([[x, y + bh], [x + bw, y + bh], [x + bw, y], [x, y]], w, h)
        else:
            continue
        out.append((text, float(conf), q))
    if size is not None and w and h and (w, h) != tuple(size):
        sx, sy = size[0] / w, size[1] / h
        out = [(t, c, q * np.array([sx, sy])) for t, c, q in out]
    return out


class JsonBackend:
    name = "json"

    def __init__(self, results_dir):
        self.dir = pathlib.Path(results_dir)

    def signature(self):
        return {"backend": self.name, "dir": str(self.dir)}

    def lookup(self, key, size=None):
        p = self.dir / f"{key}.json"
        if not p.exists():
            raise FileNotFoundError(f"no precomputed OCR for {key!r} in {self.dir}")
        with open(p) as f:
            return parse_json_result(json.load(f), size)


_BACKEND_CACHE = {}


def get_backend(name, **kwargs):
    """Backend instance, cached per (name, kwargs) so models load once per process."""
    key = (name, json.dumps(kwargs, sort_keys=True, default=str))
    if key not in _BACKEND_CACHE:
        if name == "vision":
            _BACKEND_CACHE[key] = VisionBackend(**kwargs)
        elif name == "paddle":
            _BACKEND_CACHE[key] = PaddleBackend(**kwargs)
        elif name == "json":
            _BACKEND_CACHE[key] = JsonBackend(**kwargs)
        else:
            raise ValueError(f"unknown OCR backend {name!r} (choose from {BACKENDS})")
    return _BACKEND_CACHE[key]


# ────────────────────────────────────────────────────────────────────────────
# Merge, order, run
# ────────────────────────────────────────────────────────────────────────────

def _candidate_score(it):
    n = sum(c.isalnum() for c in it["text"])
    s = it["conf"] + 0.001 * min(n, 50)
    # a multi-character reading much shorter than it is tall read across the glyphs
    if n >= 3 and quad_length(it["quad"]) < 0.8 * quad_height(it["quad"]):
        s -= 0.25
    if it["rot"] == 0:
        s += 0.02
    return s


def _pick_reading(cluster):
    """Among readings of the same line from different rotations, keep the one that agrees best
    with the others (conf + mean NED to the rest). Vision is largely orientation-robust, so
    rotated passes mostly give extra readings of the same line and this acts as a vote."""
    if len(cluster) == 1:
        return cluster[0]
    norm = [normalize(c["text"]) for c in cluster]
    best, best_s = cluster[0], None
    for i, c in enumerate(cluster):
        agree = sum(ned(norm[i], norm[j]) for j in range(len(cluster)) if j != i) / (len(cluster) - 1)
        s = c["conf"] + agree
        if best_s is None or s > best_s + 1e-9:
            best, best_s = c, s
    return best


def merge_rotations(items, iou_thr=0.3, contain_thr=0.6, same_line_iou=0.5):
    """NMS over axis-aligned quad boxes, only between items from different rotations
    (a single engine pass does not return duplicates). Suppressed items that overlap their
    keeper with IoU > same_line_iou are treated as other readings of that line (_pick_reading)."""
    order = sorted(items, key=_candidate_score, reverse=True)
    clusters = []
    for it in order:
        owner = None
        for cl in clusters:
            k = cl[0]
            if k["rot"] == it["rot"]:
                continue
            iou = box_iou(k["bbox"], it["bbox"])
            if iou > iou_thr or box_containment(k["bbox"], it["bbox"]) > contain_thr:
                owner = cl
                if iou > same_line_iou and all(m["rot"] != it["rot"] for m in cl):
                    cl.append(it)
                break
        if owner is None:
            clusters.append([it])
    return [_pick_reading(cl) for cl in clusters]


def sort_reading_order(items):
    """Top-to-bottom rows (items overlapping the row anchor by >= 50% of the smaller height),
    left-to-right inside a row."""
    rest = sorted(items, key=lambda it: (it["bbox"][1] + it["bbox"][3]) / 2)
    rows = []
    for it in rest:
        y0, y1 = it["bbox"][1], it["bbox"][3]
        if rows:
            a = rows[-1][0]["bbox"]
            ov = min(y1, a[3]) - max(y0, a[1])
            if ov >= 0.5 * min(y1 - y0, a[3] - a[1]):
                rows[-1].append(it)
                continue
        rows.append([it])
    return [it for row in rows for it in sorted(row, key=lambda it: it["bbox"][0])]


def _cache_key(img, sig, opts):
    h = hashlib.sha1()
    a = np.asarray(img)
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    h.update(json.dumps([sig, opts], sort_keys=True, default=str).encode())
    return h.hexdigest()


def ocr_images(images, backend="vision", rotations=(0,), upscale_to=None, max_upscale=4.0,
               min_conf=0.0, keys=None, cache_dir=None, backend_kwargs=None):
    """OCR several images in one backend call. Returns one sorted item list per image.

    images: paths, PIL images or arrays (alpha composited over white).
    keys:   per-image lookup keys for the json backend (default: image stem).
    """
    backend_kwargs = backend_kwargs or {}
    be = backend if hasattr(backend, "signature") else get_backend(backend, **backend_kwargs)
    rotations = tuple(int(r) % 360 for r in rotations) or (0,)
    names = [pathlib.Path(im).stem if isinstance(im, (str, os.PathLike)) else None for im in images]
    pil = [load_rgb(im) for im in images]
    results = [None] * len(pil)

    if be.name == "json":
        for i, im in enumerate(pil):
            key = (keys[i] if keys else None) or names[i]
            raw = be.lookup(key, im.size)
            items = [make_item(t, c, q) for t, c, q in raw]
            items = [it for it in items if it["conf"] >= min_conf and it["text"].strip()]
            results[i] = sort_reading_order(items)
        return results

    opts = {"rotations": rotations, "upscale_to": upscale_to, "max_upscale": max_upscale, "min_conf": min_conf}
    todo = []
    for i, im in enumerate(pil):
        if cache_dir:
            ck = _cache_key(im, be.signature(), opts)
            cp = pathlib.Path(cache_dir) / f"{ck}.json"
            if cp.exists():
                with open(cp) as f:
                    results[i] = json.load(f)
                continue
        todo.append(i)

    variants, owners = [], []
    for i in todo:
        im = pil[i]
        f = upscale_factor(im.size, upscale_to, max_upscale)
        up = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS) if f > 1 else im
        arr = np.asarray(up)
        scale = np.array([up.width / im.width, up.height / im.height])
        for rot in rotations:
            v = Image.fromarray(np.ascontiguousarray(np.rot90(arr, rot // 90))) if rot else up
            variants.append(v)
            owners.append((i, rot, scale, up.width, up.height))
    raw = be.recognize_batch(variants) if variants else []

    per_image = {i: [] for i in todo}
    for (i, rot, scale, uw, uh), found in zip(owners, raw):
        for text, conf, quad in found:
            if conf < min_conf or not str(text).strip():
                continue
            q = unrotate_points(quad, rot, uw, uh) / scale
            per_image[i].append(make_item(text, conf, q, rot))
    for i in todo:
        items = per_image[i]
        if len(rotations) > 1:
            items = merge_rotations(items)
        results[i] = sort_reading_order(items)
        if cache_dir:
            ck = _cache_key(pil[i], be.signature(), opts)
            pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(pathlib.Path(cache_dir) / f"{ck}.json", "w") as f:
                json.dump(results[i], f)
    return results


def ocr_image(image, backend="vision", **kwargs):
    """Single-image convenience wrapper around ocr_images (key= for the json backend)."""
    key = kwargs.pop("key", None)
    return ocr_images([image], backend, keys=[key] if key else None, **kwargs)[0]


def backend_kwargs_from_args(args):
    """Backend constructor kwargs from the shared CLI flags (add_ocr_args)."""
    if args.backend == "vision":
        kw = {}
        if args.langs:
            kw["langs"] = args.langs.split(",")
        if args.vision_correction:
            kw["correction"] = True
        return kw
    if args.backend == "paddle":
        kw = {"lang": args.paddle_lang}
        if args.paddle_det_limit:
            kw["det_limit_side_len"] = args.paddle_det_limit
        if args.paddle_device:
            kw["device"] = args.paddle_device
        return kw
    if not args.ocr_json_dir:
        raise SystemExit("--backend json needs --ocr-json-dir")
    return {"results_dir": args.ocr_json_dir}


def add_ocr_args(p, default_backend=None):
    """Shared OCR flags for the CLIs in this package (default backend: vision on macOS, else paddle)."""
    default_backend = default_backend or ("vision" if platform.system() == "Darwin" else "paddle")
    p.add_argument("--backend", choices=BACKENDS, default=default_backend)
    p.add_argument("--langs", default=None, help="vision: comma list, e.g. en-US,es-ES")
    p.add_argument("--vision-correction", action="store_true", help="vision: enable language correction")
    p.add_argument("--paddle-lang", default="en")
    p.add_argument("--paddle-det-limit", type=int, default=None,
                   help="paddle: detector max side (2.x default here 2048, 3.x keeps its default)")
    p.add_argument("--paddle-device", default=None, help="paddle: gpu / cpu (default: auto)")
    p.add_argument("--ocr-json-dir", default=None, help="json backend: dir of <key>.json results")
    return p


def parse_rotations(s):
    return tuple(int(x) for x in str(s).split(",") if x.strip() != "")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("images", nargs="+")
    add_ocr_args(p)
    p.add_argument("--rotations", default="0", help="comma list of CCW degrees, e.g. 0,90,270")
    p.add_argument("--upscale-to", type=int, default=None, help="upscale so the longest side reaches this")
    p.add_argument("--max-upscale", type=float, default=4.0)
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--key", default=None, help="json backend lookup key (single image)")
    p.add_argument("--out", default=None, help="write results JSON here")
    args = p.parse_args(argv)

    res = ocr_images(args.images, args.backend, rotations=parse_rotations(args.rotations),
                     upscale_to=args.upscale_to, max_upscale=args.max_upscale, min_conf=args.min_conf,
                     keys=[args.key] if args.key else None, backend_kwargs=backend_kwargs_from_args(args))
    for path, items in zip(args.images, res):
        print(f"{path}: {len(items)} lines")
        for it in items:
            print(f"  {it['conf']:.2f} rot={it['rot']:3d} h={it['height']:6.1f}  {it['text']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"backend": args.backend, "results": dict(zip(args.images, res))}, f, indent=1)


if __name__ == "__main__":
    main()
