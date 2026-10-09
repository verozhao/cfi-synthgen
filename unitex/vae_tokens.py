"""
VAE token check: do a line's glyph tokens look like the photo's tokens at the spot they claim?

For every front-view line of a registered layout (text_reg.json with front_image, anchors.py
--register-run --front-image), the patch of each glyph kind (fixed, box, gt) is encoded with the
FLUX VAE the way the pipeline does it (glyph_tokens.encode_patches, posterior mode) and packed into
64-number tokens. The registered photo is cut at the same front-view positions (the tokens' row and
column ids, one 16 x 16 px cell per token) and encoded the same way. Then:

  matched   mean cosine similarity between glyph token k and the photo token at its position
  baseline  mean cosine over all glyph x photo token pairs of the line, positions ignored
  edges     matched cosine for Canny edge maps of the patch and of the photo cut: the input an edge
            map condition would give, which ignores colour and lighting

gt patches are cut from the same photo, so their matched cosine is close to 1. It is below 1 because
a patch is its box resampled to whole tokens while its position ids snap to whole tokens around the
box centre, so a token can sit up to half a token off the photo cell it is compared with.
Outputs in --out: vae_tokens.json (per line, mean per kind, summary_html and figures_html for
stage4_page.py --vae-json) and latents_<line>.png (each patch, its latent's first three principal
components as RGB, and the VAE's reconstruction).

Usage (thanos6, CPU is enough for one product):
  python -m unitex.vae_tokens --eval-dir E --sku 016000233164 --text text_reg.json \\
      --vae <FLUX.1-dev dir or hub id> --view-res 1024 --out <dir>
"""

import argparse
import base64
import html
import io
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from unitex import glyph as gl
from unitex import glyph_tokens as gt
from unitex.common import TOKEN_PX

KINDS = ("fixed", "box", "gt")


def footprint_px(inst):
    """(x0, y0, x1, y1) front-view px of the instance's token grid (ids are token centres - 0.5)."""
    ids = np.asarray(inst.ids)
    r0, c0 = int(round(ids[:, 1].min())), int(round(ids[:, 2].min()))
    th, tw = inst.token_hw
    return c0 * TOKEN_PX, r0 * TOKEN_PX, (c0 + tw) * TOKEN_PX, (r0 + th) * TOKEN_PX


def cut(front, box, fill=128):
    """front (R, R, 3) uint8 cut at box, padded with grey where the box leaves the view."""
    x0, y0, x1, y1 = box
    out = np.full((y1 - y0, x1 - x0, 3), fill, np.uint8)
    R = front.shape[0]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(R, x1), min(R, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = front[sy0:sy1, sx0:sx1]
    return Image.fromarray(out)


def edges(im):
    import cv2
    g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    v = float(np.median(g))
    e = cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)) or 1)
    return Image.fromarray(np.repeat(e[..., None], 3, axis=2))


def tokens(latent):
    return gt.pack_latents(latent.float())[0]                    # [n, 64]


def cosines(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return float((a * b).sum(-1).mean()), float((a @ b.T).mean())


def pca_rgb(latents):
    """Latents [1, C, h, w] -> RGB images of their first three principal components (shared basis)."""
    flat = [z[0].float().reshape(z.shape[1], -1).T for z in latents]
    X = torch.cat(flat)
    X = X - X.mean(0)
    _, _, V = torch.pca_lowrank(X, q=3, center=False)
    out = []
    for z, f in zip(latents, flat):
        p = (f - X.mean(0)) @ V
        p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values + 1e-6)
        out.append(Image.fromarray((p.reshape(z.shape[2], z.shape[3], 3).numpy() * 255).astype(np.uint8)))
    return out


def decode(vae, z):
    with torch.no_grad():
        x = vae.decode(z / vae.config.scaling_factor + (vae.config.shift_factor or 0.0)).sample
    a = x[0].float().clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(a)


def _uri(im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sheet(rows, scale=2):
    """rows: [(label, [PIL images])] -> one image, every image upscaled by `scale` (nearest)."""
    ims = [[im.resize((im.width * scale, im.height * scale), Image.NEAREST) for im in r] for _, r in rows]
    W = max(sum(i.width for i in r) + 8 * (len(r) - 1) for r in ims)
    H = sum(max(i.height for i in r) for r in ims) + 8 * (len(ims) - 1)
    out = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in ims:
        x = 0
        for im in r:
            out.paste(im.convert("RGB"), (x, y))
            x += im.width + 8
        y += max(i.height for i in r) + 8
    return out


def run(text_path, vae, view_res, out_dir, device="cpu"):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = gt.load_text(str(text_path))
    if not doc.get("front_image"):
        raise SystemExit(f"{text_path} has no front_image (anchors.py --register-run --front-image)")
    front = gt.front_view_image(doc["front_image"], doc.get("_dir"), view_res)
    insts = {k: [i for i in gt.build_infer_glyphs(str(text_path), gl.GlyphConfig(infer_kind=k), view_res=view_res)
                 if i.raw_view == 0] for k in KINDS}
    lines, figures = [], []
    ids = sorted(set.intersection(*[{i.item_id for i in v} for v in insts.values()]))
    for item_id in ids:
        row = {"item_id": item_id}
        lat_all, ims_all = [], []
        for k in KINDS:
            inst = next(i for i in insts[k] if i.item_id == item_id)
            row["text"] = inst.text
            photo = cut(front, footprint_px(inst))
            pz, phz, pe, phe = gt.encode_patches([inst.patch.convert("RGB"), photo, edges(inst.patch), edges(photo)],
                                                 vae, "mode", device=device)
            m, b = cosines(tokens(pz), tokens(phz))
            me, _ = cosines(tokens(pe), tokens(phe))
            row[k] = {"tokens": int(tokens(pz).shape[0]), "matched": round(m, 4), "baseline": round(b, 4),
                      "edges_matched": round(me, 4), "footprint_px": list(footprint_px(inst))}
            lat_all += [pz, phz]
            ims_all += [(f"{k} patch", inst.patch.convert("RGB")), (f"photo at its {k} tokens", photo)]
        pcs = pca_rgb(lat_all)
        rows = [(lab, [im, pc.resize(im.size, Image.NEAREST), decode(vae, z)])
                for (lab, im), pc, z in zip(ims_all, pcs, lat_all)]
        sh = sheet(rows, scale=1)
        safe = "".join(c if c.isalnum() else "_" for c in row["text"])[:24]
        sh.save(out_dir / f"latents_{item_id:02d}_{safe}.png")
        figures.append((row["text"], sh))
        lines.append(row)
    mean = {k: {m: round(float(np.mean([r[k][m] for r in lines])), 4) for m in ("matched", "baseline", "edges_matched")}
            for k in KINDS} if lines else {}
    trs = "".join(f"<tr><td>{html.escape(r['text'])}</td>" + "".join(
        f"<td>{r[k]['matched']:.2f} / {r[k]['baseline']:.2f} / {r[k]['edges_matched']:.2f}</td>" for k in KINDS)
        + "</tr>" for r in lines)
    summary = ("Each cell: matched / baseline / edges (cosine similarity of 64-number VAE tokens). Matched compares "
               "every glyph token with the photo's token at the front-view spot it claims. Baseline ignores "
               "positions. gt is cut from the photo itself: its matched value is below 1 only because tokens "
               "snap to a 16 px grid around the box centre."
               f'<table class="pt"><tr><th>line</th>{"".join(f"<th>{k}</th>" for k in KINDS)}</tr>{trs}'
               "<tr><td><b>mean</b></td>" + "".join(
                   f"<td><b>{mean[k]['matched']:.2f} / {mean[k]['baseline']:.2f} / {mean[k]['edges_matched']:.2f}</b></td>"
                   for k in KINDS) + "</tr></table>") if lines else "No front-view line had all three kinds."
    figs = "".join(f'<figure style="flex:0 1 640px"><img src="{_uri(im)}" data-caption="{html.escape(t)}">'
                   f'<figcaption>{html.escape(t)}: patch, latent (3 main components as colour), VAE reconstruction'
                   f'</figcaption></figure>' for t, im in figures[:4])
    res = {"text": str(text_path), "view_res": view_res, "lines": lines, "mean": mean, "summary_html": summary,
           "figures_html": figs}
    with open(out_dir / "vae_tokens.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(mean, indent=1))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--sku", required=True)
    p.add_argument("--text", default="text_reg.json")
    p.add_argument("--vae", required=True, help="FLUX.1-dev directory (vae/ inside) or hub id")
    p.add_argument("--view-res", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae", torch_dtype=torch.float32).to(args.device).eval()
    text = pathlib.Path(args.text)
    text = text if text.is_absolute() else pathlib.Path(args.eval_dir) / args.sku / text
    return run(text, vae, args.view_res, args.out, args.device)


if __name__ == "__main__":
    main()
