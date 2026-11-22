"""
Per-stage times of UniTEX runs, read from their console logs.

run_unitex.py and its variants print UniTEX's own timers ('>>> name seconds >>>'), one progress bar per FLUX
pass (the texture pass, then the delight pass) and one '[sku] ok in N s, peak G GB' line per product. This prints
the median of each stage over the products in the given logs and its share of a product.

Usage: python unitex/stage_times.py LOG [LOG ...]        (all shards of one run)
"""

import re
import statistics
import sys

STAGE = re.compile(r">>> (\S+(?: \S+)*?) ([0-9.]+) >>>")
TQDM = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")
OK = re.compile(r"\[(\d+)\] ok in ([0-9.]+) s, peak ([0-9.]+) GB")
INNER = ("render", "inverse_render", "image_to_uv_v1", "image_to_uv_v2", "uv_to_image",   # inside the bake's timer
         "texture_pass", "delight_pass")                                                  # inside infer_mv's


def parse(text):
    """Log text -> one dict per finished product: sku, wall, peak, passes [(steps, seconds)], stages {name: [s]}."""
    rows = []
    cur = {"passes": [], "stages": {}}
    last = None
    for line in text.replace("\r", "\n").split("\n"):
        m = TQDM.search(line)
        if m and "it" in line and "Fetching" not in line and "Loading" not in line:
            n, tot = int(m.group(1)), int(m.group(2))
            a, b, c = int(m.group(3)), int(m.group(4)), m.group(5)
            sec = a * 3600 + b * 60 + int(c) if c else a * 60 + b
            if n == tot and tot >= 2:
                if last != (tot, sec):              # tqdm prints the full bar twice
                    cur["passes"].append((tot, sec))
                last = (tot, sec)
            else:
                last = None
            continue
        m = STAGE.search(line)
        if m:
            cur["stages"].setdefault(m.group(1), []).append(float(m.group(2)))
            continue
        m = OK.search(line)
        if m:
            cur.update(sku=m.group(1), wall=float(m.group(2)), peak=float(m.group(3)))
            rows.append(cur)
            cur = {"passes": [], "stages": {}}
            last = None
    return rows


def summary(rows):
    """[(label, median seconds)] in log order, the two passes first, and the median product time."""
    med = statistics.median
    wall = med(r["wall"] for r in rows)
    out = []
    for k, label in ((0, "texture pass (FLUX)"), (1, "delight pass (FLUX)")):
        xs = [r["passes"][k] for r in rows if len(r["passes"]) > k]
        if xs:
            steps, sec = xs[0][0], med(x[1] for x in xs)
            out.append((f"{label}, {steps} steps, {sec / steps:.1f} s per step", sec))
    flux = sum(s for _, s in out)
    names = []
    for r in rows:
        names += [k for k in r["stages"] if k not in names]
    for k in names:
        if k.startswith("texture:") or k in INNER:
            continue
        xs = [sum(r["stages"][k]) for r in rows if k in r["stages"]]
        cnt = int(med(len(r["stages"][k]) for r in rows if k in r["stages"]))
        if k == "infer_mv":
            out.append(("infer_mv, other than the two passes", med(xs) - flux))
        else:
            out.append((f"{k} (x{cnt})" if cnt > 1 else k, med(xs)))
    out.append(("not in any timer (saving files, GLB export)", wall - sum(s for _, s in out)))
    return out, wall


def main(argv=None):
    paths = sys.argv[1:] if argv is None else argv
    rows = []
    for p in paths:
        rows += parse(open(p, errors="replace").read())
    print(f"{len(rows)} products")
    if not rows:
        return 1
    out, wall = summary(rows)
    peak = statistics.median(r["peak"] for r in rows)
    print(f"one product            {wall:7.1f} s  ({wall / 60:.1f} min), peak GPU memory {peak:.1f} GB")
    for label, s in out:
        print(f"  {label:58s} {s:7.1f} s  {100 * s / wall:5.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
