"""
Back panel: true, typeset content on the back of a finished UniTEX run, then re-bake.

With only a front photo the back is unknown, and the texture model fills it with lines that look like print but
read as nothing. This keeps the generated back's colours (a wide blur of it, as photo front's detail mode keeps the
generated front's) and places, inside the part of the surface that faces the back camera:
  - a Nutrition Facts panel in the US format from the product's Open Food Facts record, per serving when the
    record has a serving size, else per 100 g, with only the rows the record has
  - the ingredients and allergens from the same record
  - a UPC-A barcode of the product's own code, and "Distributed by <brand>" when the record names a brand
A product without a record gets the barcode only, so nothing on the back is made up. Then only UniTEX's bake
runs again (no FLUX), as in photo_front.py.

Outputs in <eval>/<sku>/<run-name>/: textured_mesh.glb, cache/mv_rgb.png (cache/mv_rgb_before_back.png is the
input), back_panel.png (generated back | weight | result), run_info.json.
  python back_panel.py --unitex-root UX --eval-dir EVAL --src-run RUN --run-name RUN_back --off-dir OFF [--skus a,b]
"""
import argparse
import datetime
import json
import math
import pathlib
import re
import shutil
import time
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unitex import anchors as A
from unitex import common as C
from unitex import photo_front as PF

BACK = 2
TO_BACK_CAMERA = np.array([0.0, 1.0, 0.0])      # RAW_C2W[2] sits at +Y looking at the origin
FONT_DIR = "/usr/share/fonts/truetype/liberation/"
INK = (18, 18, 18)
PAPER = (255, 255, 255)


def font(size, bold=False, narrow=False):
    names = (["LiberationSansNarrow-Bold.ttf", "LiberationSans-Bold.ttf"] if narrow and bold else
             ["LiberationSansNarrow-Regular.ttf", "LiberationSans-Regular.ttf"] if narrow else
             ["LiberationSans-Bold.ttf"] if bold else ["LiberationSans-Regular.ttf"])
    for n in names:
        for d in (FONT_DIR, "/usr/share/fonts/truetype/liberation2/"):
            p = pathlib.Path(d) / n
            if p.exists():
                return ImageFont.truetype(str(p), max(4, int(size)))
    return ImageFont.load_default()


def fit_font(draw, text, max_w, size, **kw):
    f = font(size, **kw)
    while size > 6 and draw.textlength(text, font=f) > max_w:
        size *= 0.94
        f = font(size, **kw)
    return f


def wrap(draw, words, f, max_w):
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cur and draw.textlength(t, font=f) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


# ── nutrition values (Open Food Facts stores masses in g) ────────────────────────────────────────

ROWS = (("Total Fat", "fat", "g", 78, True, 0), ("Saturated Fat", "saturated-fat", "g", 20, False, 1),
        ("Trans Fat", "trans-fat", "g", None, False, 1), ("Cholesterol", "cholesterol", "mg", 300, True, 0),
        ("Sodium", "sodium", "mg", 2300, True, 0), ("Total Carbohydrate", "carbohydrates", "g", 275, True, 0),
        ("Dietary Fiber", "fiber", "g", 28, False, 1), ("Total Sugars", "sugars", "g", None, False, 1),
        ("Includes {} Added Sugars", "added-sugars", "g", 50, False, 2), ("Protein", "proteins", "g", None, True, 0))
VITS = (("Vitamin D", "vitamin-d", "mcg", 20), ("Calcium", "calcium", "mg", 1300), ("Iron", "iron", "mg", 18),
        ("Potassium", "potassium", "mg", 4700))
SCALE = {"g": 1.0, "mg": 1e3, "mcg": 1e6}


def fda_round(v, unit, key):
    """Label rounding (21 CFR 101.9, simplified). Returns the printed amount without the unit."""
    if key in ("fat", "saturated-fat", "trans-fat"):
        return "0" if v < 0.5 else (f"{round(v * 2) / 2:g}" if v < 5 else f"{round(v):d}")
    if key == "cholesterol":
        return "0" if v < 2 else ("<5" if v < 5 else f"{5 * round(v / 5):d}")
    if key == "sodium":
        return "0" if v < 5 else (f"{5 * round(v / 5):d}" if v <= 140 else f"{10 * round(v / 10):d}")
    if unit == "g":
        return "0" if v < 0.5 else ("<1" if v < 1 else f"{round(v):d}")
    return f"{v:.1f}".rstrip("0").rstrip(".") if v < 10 else f"{round(v):d}"


def kcal_round(v):
    return 0 if v < 5 else (5 * round(v / 5) if v <= 50 else 10 * round(v / 10))


def grams(text):
    """First mass in a string like '1 cup (36 g)' or '18.6 oz (527g)', in g."""
    if not text:
        return None
    m = re.search(r"([\d.]+)\s*g\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*oz\b", text, re.I)
    return float(m.group(1)) * 28.3495 if m else None


def clean_serving(s):
    """'Amount per serving (318 g)' -> '318 g', '2 tbsp (33 g) (33 g)' -> '2 tbsp (33 g)'."""
    if not s:
        return s
    s = re.sub(r"(?i)^\s*amount per serving\s*", "", str(s)).strip()
    s = re.sub(r"(\([^)]*\))(\s*\1)+", r"\1", s)
    s = re.sub(r"\d+\.\d{2,}", lambda m: f"{float(m.group(0)):.0f}", s)
    return s[1:-1] if s.startswith("(") and s.endswith(")") and s.count("(") == 1 else s


def nutrition_rows(rec):
    n = rec.get("nutriments", {})
    rec = dict(rec, serving_size=clean_serving(rec.get("serving_size")))
    per_serving = bool(rec.get("serving_size")) and "energy-kcal_serving" in n
    sfx = "_serving" if per_serving else "_100g"
    if "energy-kcal" + sfx not in n:
        return None
    rows = []
    for label, key, unit, dv, bold, indent in ROWS:
        if key + sfx not in n:
            continue
        v = float(n[key + sfx]) * SCALE[unit]
        amount = fda_round(v, unit, key) + unit
        pct = f"{round(100 * v / dv):d}%" if (dv and per_serving) else ""
        rows.append((label.format(amount) if "{}" in label else label, "" if "{}" in label else amount, pct, bold, indent))
    vits = []
    for label, key, unit, dv in VITS:
        if key + sfx in n:
            v = float(n[key + sfx]) * SCALE[unit]
            vits.append((label, fda_round(v, unit, key) + unit, f"{round(100 * v / dv):d}%" if per_serving else ""))
    servings = None
    sg, qg = grams(rec.get("serving_size")), grams(rec.get("quantity"))
    if per_serving and sg and qg and qg > sg:
        servings = qg / sg
    return {"per_serving": per_serving, "serving": rec.get("serving_size"), "servings": servings,
            "kcal": kcal_round(float(n["energy-kcal" + sfx])), "rows": rows, "vits": vits}


def nutrition_panel(nut, width):
    """US Nutrition Facts panel as an RGB image of the given width (drawn at 600 px and resized)."""
    W, pad = 600, 14
    img = Image.new("RGB", (W, 1600), PAPER)
    d = ImageDraw.Draw(img)
    inner = W - 2 * pad
    y = pad + 2

    def rule(h):
        nonlocal y
        d.rectangle([pad, y, W - pad, y + h - 1], fill=INK)
        y += h + 4

    f = fit_font(d, "Nutrition Facts", inner, 70, bold=True, narrow=True)
    d.text((pad, y - 6), "Nutrition Facts", font=f, fill=INK)
    y += f.size + 2
    rule(2)
    if nut["servings"]:
        s = nut["servings"]
        txt = f"About {s:.0f} servings per container" if abs(s - round(s)) > 0.15 else f"{round(s):d} servings per container"
        d.text((pad, y), txt, font=font(25), fill=INK)
        y += 32
    if nut["per_serving"]:
        fb = font(29, bold=True)
        d.text((pad, y), "Serving size", font=fb, fill=INK)
        sv = str(nut["serving"])
        fs = fit_font(d, sv, inner - d.textlength("Serving size", font=fb) - 12, 29, bold=True)
        d.text((W - pad - d.textlength(sv, font=fs), y), sv, font=fs, fill=INK)
        y += 36
    rule(16)
    d.text((pad, y), "Amount per serving" if nut["per_serving"] else "Amount per 100 g", font=font(21, bold=True), fill=INK)
    y += 24
    fc = font(50, bold=True)
    d.text((pad, y), "Calories", font=fc, fill=INK)
    kc = str(nut["kcal"])
    fk = font(60, bold=True)
    d.text((W - pad - d.textlength(kc, font=fk), y - 8), kc, font=fk, fill=INK)
    y += 60
    rule(8)
    if nut["per_serving"]:
        t = "% Daily Value*"
        fd = font(20, bold=True)
        d.text((W - pad - d.textlength(t, font=fd), y), t, font=fd, fill=INK)
        y += 26
    for label, amount, pct, bold, indent in nut["rows"]:
        d.rectangle([pad + (24 if indent else 0), y, W - pad, y], fill=INK)
        y += 4
        x = pad + 24 * indent
        fl = font(23, bold=bold)
        d.text((x, y), label, font=fl, fill=INK)
        if amount:
            d.text((x + d.textlength(label, font=fl) + 8, y), amount, font=font(23), fill=INK)
        if pct:
            fp = font(23, bold=True)
            d.text((W - pad - d.textlength(pct, font=fp), y), pct, font=fp, fill=INK)
        y += 30
    if nut["vits"]:
        rule(16)
        for i, (label, amount, pct) in enumerate(nut["vits"]):
            if i:
                d.rectangle([pad, y, W - pad, y], fill=INK)
                y += 4
            fv = font(22)
            d.text((pad, y), f"{label} {amount}", font=fv, fill=INK)
            if pct:
                d.text((W - pad - d.textlength(pct, font=fv), y), pct, font=fv, fill=INK)
            y += 29
    rule(8)
    if nut["per_serving"]:
        foot = ("* The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a daily "
                "diet. 2,000 calories a day is used for general nutrition advice.")
        ff = font(17)
        for line in wrap(d, foot.split(), ff, inner):
            d.text((pad, y), line, font=ff, fill=INK)
            y += 21
    y += pad
    img = img.crop((0, 0, W, y))
    d = ImageDraw.Draw(img)
    d.rectangle([1, 1, W - 2, y - 2], outline=INK, width=4)
    h = max(1, round(y * width / W))
    return img.resize((int(width), h), Image.LANCZOS)


# ── barcode ──────────────────────────────────────────────────────────────────────────────────────

L_CODE = ("0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111",
          "0001011")


def upc_valid(code):
    if not re.fullmatch(r"\d{12}", code):
        return False
    d = [int(c) for c in code]
    return (10 - (3 * sum(d[0:11:2]) + sum(d[1:11:2])) % 10) % 10 == d[11]


def upc_bits(code):
    """The 95 modules of a UPC-A code: guard, six L digits, centre guard, six R digits (L inverted), guard."""
    left = "".join(L_CODE[int(c)] for c in code[:6])
    right = "".join("".join("1" if b == "0" else "0" for b in L_CODE[int(c)]) for c in code[6:])
    return "101" + left + "01010" + right + "101"


def barcode(code, width, height):
    """UPC-A barcode with its digits, on white, as an RGB image."""
    bits = upc_bits(code)
    long_bars = set(range(0, 10)) | set(range(45, 50)) | set(range(85, 95))   # guards, first and last digit
    quiet = 11
    S = 4                                            # drawing scale, resized at the end
    mod = S * 3
    W = (95 + 2 * quiet) * mod
    H = int(W * height / width)
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    digit_h = int(H * 0.2)
    bar_h = H - digit_h - 2 * mod
    for i, b in enumerate(bits):
        if b == "1":
            x = (quiet + i) * mod
            extra = digit_h // 2 if i in long_bars else 0
            d.rectangle([x, mod, x + mod - 1, mod + bar_h + extra], fill=INK)
    f = font(digit_h * 0.9)
    ytxt = mod + bar_h + 2
    d.text((quiet * mod - d.textlength(code[0], font=font(digit_h * 0.7)) - mod, ytxt + digit_h * 0.15), code[0],
           font=font(digit_h * 0.7), fill=INK)
    for grp, x0 in ((code[1:6], quiet + 10), (code[6:11], quiet + 50)):
        t = grp
        x = (x0 + 17.5) * mod - d.textlength(t, font=f) / 2
        d.text((x, ytxt), t, font=f, fill=INK)
    d.text(((quiet + 95) * mod + mod, ytxt + digit_h * 0.15), code[11], font=font(digit_h * 0.7), fill=INK)
    return img.resize((int(width), int(height)), Image.LANCZOS)


# ── text blocks ──────────────────────────────────────────────────────────────────────────────────

ALLERGENS = {"en:milk": "MILK", "en:eggs": "EGG", "en:soybeans": "SOY", "en:peanuts": "PEANUTS",
             "en:nuts": "TREE NUTS", "en:fish": "FISH", "en:crustaceans": "SHELLFISH", "en:sesame-seeds": "SESAME"}


def ingredients_text(rec):
    ing = (rec.get("ingredients_text_en") or rec.get("ingredients_text") or "").strip()
    if not ing:
        return None
    ing = re.sub(r"\s+", " ", ing).rstrip(".").upper() + "."
    tags = rec.get("allergens_tags") or []
    names = [ALLERGENS[t] for t in tags if t in ALLERGENS]
    if "en:gluten" in tags:
        names.insert(0, "WHEAT" if "WHEAT" in ing else "GLUTEN")
    return ing, ("CONTAINS " + ", ".join(dict.fromkeys(names)) + " INGREDIENTS.") if names else None


def text_box(paragraphs, width, max_h, size=24):
    """White box with bold-led paragraphs [(bold head, body)], font shrunk until it fits max_h. RGB image."""
    W = 900
    while True:
        img = Image.new("RGB", (W, 4000), PAPER)
        d = ImageDraw.Draw(img)
        pad, y = int(size * 0.7), int(size * 0.6)
        fb, fr = font(size, bold=True), font(size)
        for head, body in paragraphs:
            words = [(w, True) for w in head.split()] + [(w, False) for w in body.split()]
            x = pad
            line_h = int(size * 1.22)
            for w, b in words:
                f = fb if b else fr
                tw = d.textlength(w + " ", font=f)
                if x + tw > W - pad and x > pad:
                    x, y = pad, y + line_h
                d.text((x, y), w, font=f, fill=INK)
                x += tw
            y += line_h + int(size * 0.4)
        y += pad // 2
        h = y * width / W
        if h <= max_h or size < 9:
            return img.crop((0, 0, W, y)).resize((int(width), max(1, int(h))), Image.LANCZOS)
        size *= 0.92


# ── layout on the back view ──────────────────────────────────────────────────────────────────────

def inner_rect(region, cover=0.985):
    """Axis-aligned rectangle mostly inside a boolean region: its 1..99 % bbox, shrunk until covered."""
    ys, xs = np.nonzero(region)
    if len(xs) < 100:
        return None
    x0, x1 = np.percentile(xs, [1, 99]).astype(int)
    y0, y1 = np.percentile(ys, [1, 99]).astype(int)
    for _ in range(40):
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        if region[y0:y1, x0:x1].mean() >= cover:
            return x0, y0, x1, y1
        dx, dy = max(1, (x1 - x0) // 50), max(1, (y1 - y0) // 50)
        x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy
    return None


def base_fill(gen, core, res, base="flat", sigma_frac=1 / 8):
    """Background for a cleaned view, from the view after a median filter of about res / 48 px, which removes
    the strokes of made-up print and keeps the label's colour areas: its wide blur (blur), or its dominant colour,
    the fullest bin of an 8-level colour histogram over core (flat)."""
    k = max(3, int(res / 48) | 1)
    med = cv2.medianBlur(np.ascontiguousarray(gen, dtype=np.uint8), k)
    dom = dominant_color(med, core)
    if dom is None:
        return gen.astype(np.float32)
    if base == "flat":
        return np.broadcast_to(dom, gen.shape).astype(np.float32).copy()
    lum_w = np.array([0.299, 0.587, 0.114], np.float32)
    if float(dom @ lum_w) < 200:                     # not a white label: white print boxes become label colour
        med = med.copy()
        med[core & (med.astype(np.float32) @ lum_w > 215)] = np.round(dom).astype(np.uint8)
    return PF.masked_blur(med.astype(np.float32), core, max(1.0, res * sigma_frac))


def dominant_color(img, region):
    """Mean of the fullest bin of an 8-level colour histogram over region, float RGB, None for an empty region."""
    px = img[region].reshape(-1, 3).astype(np.int64)
    if len(px) == 0:
        return None
    keys = (px[:, 0] // 32) * 64 + (px[:, 1] // 32) * 8 + px[:, 2] // 32
    return px[keys == np.bincount(keys).argmax()].mean(axis=0).astype(np.float32)


def clean_front_edges(view0, mask0, normals0, w_photo, res, band=(0.35, 0.55)):
    """Flat colour on the band of the front view that turns away from the camera (cos to the front camera
    below band[1], fully below band[0]) where photo front did not use the photo. A bake cutoff past 60 deg takes
    this band from the front view, whose generated pixels there are stretched round the curve."""
    cos = normals0 @ PF.TO_FRONT_CAMERA
    f = np.clip((band[1] - cos) / (band[1] - band[0]), 0.0, 1.0)
    w = f * f * (3 - 2 * f) * (1.0 - np.clip(w_photo, 0.0, 1.0))
    w[~mask0] = 0.0
    ring = mask0 & (cos > 0.2) & (cos < band[1])
    if not ring.any():
        return view0
    base = base_fill(view0, ring, res, "flat")
    out = w[..., None] * base + (1 - w[..., None]) * view0.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def compose_back(gen2, mask2, normals2, rec, upc, res, sigma_frac=1 / 8, facing=(0.05, 0.25), base_mode="flat"):
    """New back view, the blend weight and what was placed."""
    feather = max(1.0, res / 128)
    r = int(math.ceil(2 * feather))
    core = cv2.erode(mask2.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    cos = normals2 @ TO_BACK_CAMERA
    f = np.clip((cos - facing[0]) / (facing[1] - facing[0]), 0.0, 1.0)
    f = f * f * (3 - 2 * f)
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), feather) * f
    w[~mask2] = 0.0
    base = base_fill(gen2, core, res, base_mode, sigma_frac)
    canvas = Image.fromarray(np.clip(np.rint(base), 0, 255).astype(np.uint8))
    rect = inner_rect(core & (cos > 0.8))
    cols = np.nonzero(mask2.any(axis=0))[0]
    if rect is not None and len(cols) and rect[2] - rect[0] < 0.75 * (cols[-1] - cols[0]):
        rect = inner_rect(core & (cos > 0.55)) or rect      # curved backs (cans): panels may wrap further round
    placed = []
    if rect is None:
        return gen2.copy(), w, ["no back-facing area"]
    x0, y0, x1, y1 = rect
    m = int(0.05 * min(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = x0 + m, y0 + m, x1 - m, y1 - m
    Wr, Hr = x1 - x0, y1 - y0
    gap = max(4, int(0.03 * min(Wr, Hr)))
    nut = nutrition_rows(rec) if rec else None
    ing = ingredients_text(rec) if rec else None
    brand = (rec.get("brands") or "").split(",")[0].strip() if rec else ""
    if brand.lower() in ("", "marca", "brand", "none", "unknown"):
        brand = ""
    name = (rec.get("product_name") or "").strip() if rec else ""
    d = ImageDraw.Draw(canvas)
    lum = float(np.mean(np.asarray(canvas)[y0:y1, x0:x1] @ [0.299, 0.587, 0.114]))
    tcol = INK if lum > 140 else (250, 250, 250)
    y = y0
    if name:
        fn = fit_font(d, name, Wr, max(10, 0.075 * Hr if Hr > Wr else 0.11 * Hr), bold=True)
        d.text((x0 + (Wr - d.textlength(name, font=fn)) / 2, y), name, font=fn, fill=tcol)
        y += int(fn.size * 1.3)
        placed.append("name")
    tall = Hr > 1.2 * Wr
    narrow = Wr < 0.55 * Hr                          # can bands: the panels get most of the width
    bw = int((0.8 if narrow else 0.62 if tall else 0.42) * Wr)
    bh = int(bw * 0.42)
    if nut:
        nw = int((0.94 if narrow else (0.72 if ing else 0.85) if tall else 0.46) * Wr)
        panel = nutrition_panel(nut, nw)
        avail = (y1 - y) - (bh + gap if tall else 0)
        if tall and ing:
            avail = int(avail * 0.66)                # leave room for the ingredients below
        if panel.height > avail:
            panel = panel.resize((max(1, int(panel.width * avail / panel.height)), int(avail)), Image.LANCZOS)
        px = x0 if not tall else x0 + (Wr - panel.width) // 2
        canvas.paste(panel, (px, y))
        placed.append("nutrition")
        if tall:
            ytop = y + panel.height + gap
            right = (x0, ytop, x1, y1 - bh - gap)
        else:
            right = (px + panel.width + gap, y, x1, y1 - bh - gap)
    else:
        right = (x0, y, x1, y1 - bh - gap)
    if ing and right[2] - right[0] > 40 and right[3] - right[1] > 30:
        paras = [("INGREDIENTS:", ing[0])] + ([("", ing[1])] if ing[1] else [])
        box = text_box(paras, right[2] - right[0], right[3] - right[1])
        canvas.paste(box, (right[0], right[1]))
        placed.append("ingredients")
    if upc_valid(upc):
        bc = barcode(upc, bw, bh)
        bx = x1 - bw
        canvas.paste(bc, (bx, y1 - bh))
        placed.append("barcode")
        if brand:
            t = f"Distributed by {brand}"
            fb = fit_font(d, t, bx - x0 - gap, max(9, 0.028 * Hr if tall else 0.045 * Hr))
            d.text((x0, y1 - fb.size * 1.3), t, font=fb, fill=tcol)
            placed.append("brand")
    comp = np.asarray(canvas).astype(np.float32)
    out = w[..., None] * comp + (1 - w[..., None]) * gen2.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), w, placed


VIEW_DIRS = {1: np.array([1.0, 0.0, 0.0]), 3: np.array([-1.0, 0.0, 0.0]), 4: np.array([0.0, 0.0, 1.0]),
             5: np.array([0.0, 0.0, -1.0])}       # to the right, left, top and bottom cameras (RAW_C2W)


def clean_view(gen, mask, normals, to_cam, res, sigma_frac=1 / 8, facing=(0.2, 0.5), base_mode="flat", fill=None):
    """The view's dominant colour (or wide blur) where the surface faces its camera: made-up print removed. With
    fill (an RGB colour) that colour is used instead, e.g. the front label's for the side views."""
    feather = max(1.0, res / 128)
    r = int(math.ceil(2 * feather))
    core = cv2.erode(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    f = np.clip((normals @ to_cam - facing[0]) / (facing[1] - facing[0]), 0.0, 1.0)
    w = cv2.GaussianBlur(core.astype(np.float32), (0, 0), feather) * f * f * (3 - 2 * f)
    w[~mask] = 0.0
    base = (np.broadcast_to(np.asarray(fill, np.float32), gen.shape).copy() if fill is not None
            else base_fill(gen, core, res, base_mode, sigma_frac))
    out = w[..., None] * base + (1 - w[..., None]) * gen.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def color_map(src_run_dir, res, sigma_frac=1 / 32):
    """Affine colour map (3x4) from the generated front to the registered photo, fitted on their blurs where photo
    front used the photo (its photo_front.png: generated front | warped photo | weight | result). Highlights and
    deep shadows of the photo are left out. None when the photo was not used."""
    p = pathlib.Path(src_run_dir) / "photo_front.png"
    if not p.exists():
        return None
    panel = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    gen0, ph, wv = panel[:, :res], panel[:, res:2 * res], panel[:, 2 * res:3 * res, 0] / 255.0
    core = wv > 0.9
    if core.sum() < 0.05 * res * res:
        return None
    s = max(1.0, res * sigma_frac)
    g = PF.masked_blur(gen0, core, s)
    q = PF.masked_blur(ph, core, s)
    lum = q @ np.array([0.299, 0.587, 0.114], np.float32)
    sel = core & (lum > 12) & (lum < 243)
    ys, xs = np.nonzero(sel)
    k = max(1, len(ys) // 20000)                        # about 20k pixel pairs
    ys, xs = ys[::k], xs[::k]
    X = np.concatenate([g[ys, xs], np.ones((len(ys), 1), np.float32)], axis=1)
    Y = q[ys, xs]
    M = np.linalg.lstsq(X, Y, rcond=None)[0].T          # 3x4
    for _ in range(2):                                  # drop the worst 10 % (photo glare, misregistration) and refit
        err = np.abs(X @ M.T - Y).sum(axis=1)
        keep = err <= np.percentile(err, 90)
        M = np.linalg.lstsq(X[keep], Y[keep], rcond=None)[0].T
    return M


def apply_color_map(img, M, detail_sigma=None):
    """img through the 3x4 colour map M. With detail_sigma only the Gaussian blur of img (sigma in px) is
    mapped and the detail above it is kept as it is: an affine map with gains under 1 also lowers the
    contrast of small print, which then reads worse."""
    a = img.astype(np.float32)
    if detail_sigma:
        low = cv2.GaussianBlur(a, (0, 0), detail_sigma)
        out = low @ M[:, :3].T + M[:, 3] + (a - low)
    else:
        out = a @ M[:, :3].T + M[:, 3]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def run(args):
    eval_dir = pathlib.Path(args.eval_dir).resolve()
    p = pathlib.Path(args.skus) if args.skus else None
    if p and p.is_file():
        skus = [l.split("#")[0].strip() for l in open(p) if l.split("#")[0].strip()]
    elif args.skus:
        skus = args.skus.split(",")
    else:
        skus = sorted(d.name for d in eval_dir.iterdir() if (d / args.src_run / "cache" / "mv_rgb.png").exists())
    off = pathlib.Path(args.off_dir)
    baker = None
    n_ok = n_err = 0
    for i, sku in enumerate(skus):
        sku_dir = eval_dir / sku
        src, dst = sku_dir / args.src_run, sku_dir / args.run_name
        rec_info = {"sku": sku, "src_run": args.src_run, "run_name": args.run_name,
                    "time": datetime.datetime.now().isoformat(timespec="seconds")}
        t0 = time.time()
        try:
            if not (src / "cache" / "mv_rgb.png").exists():
                raise FileNotFoundError(f"{src}/cache/mv_rgb.png")
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "run_info.json").unlink(missing_ok=True)
            PF.copy_cache(src / "cache", dst / "cache")
            geo = A.load_unitex_cache(dst / "cache")
            res = geo.res
            grid = np.array(Image.open(dst / "cache" / "mv_rgb.png").convert("RGB"))
            M = color_map(src, res) if args.color_match else None
            if M is not None:                            # all six views, before the back and the clean views
                views = C.split_grid(grid, res)
                for v in range(6):
                    grid = PF.put_view(grid, v, apply_color_map(np.ascontiguousarray(views[v]), M, res / 64), res)
                rec_info["color_map"] = np.round(M, 4).tolist()
            gen2 = np.ascontiguousarray(C.split_grid(grid, res)[BACK])
            f = off / f"{sku}.json"
            rec = json.loads(f.read_text()) if f.exists() else None
            new2, w, placed = compose_back(gen2, geo.masks[BACK], geo.normals[BACK], rec, sku, res,
                                           args.sigma_frac, tuple(args.facing), args.back_base)
            shutil.copy(dst / "cache" / "mv_rgb.png", dst / "cache" / "mv_rgb_before_back.png")
            out_grid = PF.put_view(grid.copy(), BACK, new2, res)
            views = C.split_grid(grid, res)
            k = max(3, int(res / 48) | 1)                # the front label's colour, for the sides
            front_fill = dominant_color(cv2.medianBlur(np.ascontiguousarray(views[0]), k), geo.masks[0])
            for v in args.clean_views:
                cv = clean_view(np.ascontiguousarray(views[v]), geo.masks[v], geo.normals[v], VIEW_DIRS[v], res,
                                args.sigma_frac, base_mode=args.base,
                                fill=front_fill if (v in (1, 3) and args.sides_from_front) else None)
                out_grid = PF.put_view(out_grid, v, cv, res)
            if args.clean_front_edges:
                pf_panel = src / "photo_front.png"
                w_photo = (np.asarray(Image.open(pf_panel).convert("L"))[:, 2 * res:3 * res].astype(np.float32) / 255.0
                           if pf_panel.exists() else np.zeros((res, res), np.float32))
                v0 = clean_front_edges(np.ascontiguousarray(C.split_grid(out_grid, res)[0]), geo.masks[0],
                                       geo.normals[0], w_photo, res)
                out_grid = PF.put_view(out_grid, 0, v0, res)
            Image.fromarray(out_grid).save(dst / "cache" / "mv_rgb.png")
            wv = np.repeat((w * 255).astype(np.uint8)[..., None], 3, axis=2)
            Image.fromarray(np.concatenate([gen2, wv, new2], axis=1)).save(dst / "back_panel.png")
            rec_info.update(view_res=res, record=bool(rec), placed=placed, sigma_frac=args.sigma_frac,
                            facing=list(args.facing))
            print(f"  [{sku}] ({i + 1}/{len(skus)}) placed {placed}", flush=True)
            if not args.no_bake:
                if baker is None:
                    baker = PF.build_baker(args.unitex_root, res, args.seed)
                import torch
                baker.generator = torch.Generator().manual_seed(args.seed)
                torch.manual_seed(args.seed)
                baker(str(dst), str(sku_dir / "ref.png"), str(sku_dir / "mesh.glb"))
            rec_info["status"] = "ok"
        except KeyboardInterrupt:
            raise
        except Exception as e:
            rec_info.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            print(f"  [{sku}] ERROR {rec_info['error']}\n" + "\n".join(rec_info["traceback"].splitlines()[-6:]),
                  flush=True)
        rec_info["wall_s"] = round(time.time() - t0, 1)
        if rec_info["status"] == "ok":
            (dst / "run_info.json").write_text(json.dumps(rec_info, indent=1))
            n_ok += 1
        else:
            n_err += 1
    print(f"Done: {n_ok} ok, {n_err} errors.")
    return n_err


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unitex-root", default=".")
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--src-run", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--off-dir", required=True, help="Open Food Facts records, <sku>.json")
    p.add_argument("--skus", default=None)
    p.add_argument("--sigma-frac", type=float, default=1 / 8, help="blur of the generated back, fraction of view_res")
    p.add_argument("--facing", type=float, nargs=2, default=(0.05, 0.25), help="cos ramp of the cleaned back")
    p.add_argument("--clean-views", type=lambda s: [int(v) for v in s.split(",") if v], default=[],
                   help="also remove made-up print from these views (1 right, 3 left, 4 top, 5 bottom), e.g. 1,3,4,5")
    p.add_argument("--base", choices=("flat", "blur"), default="flat",
                   help="background of the --clean-views: dominant colour (flat) or wide blur")
    p.add_argument("--back-base", choices=("flat", "blur"), default="blur",
                   help="background of the back behind the panels: wide blur (keeps the label's colour areas) or flat")
    p.add_argument("--sides-from-front", action="store_true",
                   help="fill the cleaned side views (1, 3) with the front label's dominant colour")
    p.add_argument("--clean-front-edges", action="store_true",
                   help="flat colour on the front view's turning-away band where photo front did not use the photo")
    p.add_argument("--color-match", action="store_true",
                   help="map all views' colours to the photo's, fitted where photo front used the photo")
    p.add_argument("--seed", type=int, default=63)
    p.add_argument("--no-bake", action="store_true", help="write mv_rgb.png and back_panel.png only (CPU)")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
