"""Contact sheet of photo_front.png panels: python pf_sheet.py <run_name> <out.jpg> [width]"""
import json, pathlib, sys
from PIL import Image, ImageDraw

R = pathlib.Path("/mnt/nvme1n1/veronica_unitex")
run, out = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
rows = []
for d in ("unitex_eval_28", "unitex_eval_uvfix"):
    for p in sorted((R / d).glob(f"*/{run}/photo_front.png")):
        info = json.load(open(p.parent / "run_info.json")) if (p.parent / "run_info.json").exists() else {}
        reg = info.get("register", {})
        label = (f"{p.parent.parent.name}  {reg.get('method')}  inliers {reg.get('n_inliers')}  "
                 f"NCC aff {reg.get('ncc_affine')} hom {reg.get('ncc_homography')}  "
                 f"photo {100 * info.get('replaced_frac_of_silhouette', 0):.0f}%")
        im = Image.open(p).convert("RGB")
        im = im.resize((W, W * im.height // im.width))
        canvas = Image.new("RGB", (W, im.height + 22), (30, 30, 30))
        canvas.paste(im, (0, 22))
        ImageDraw.Draw(canvas).text((6, 4), label, fill=(255, 255, 255))
        rows.append(canvas)
sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (0, 0, 0))
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out, quality=88)
print(out, len(rows), "panels", sheet.size)
