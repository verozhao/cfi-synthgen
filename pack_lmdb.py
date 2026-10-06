"""
mvgen PNG directories -> LMDB envs for UniTEX-FLUX (launch.py reads image_ext='.mdb')

Steps:
  1. find every <root>/{render,render_random}/<volume>/<uid>/ directory holding image files
  2. write one LMDB env per directory: key = file basename without extension, value = file bytes
  3. re-open the env and verify every value byte for byte, then optionally delete the images

Only image files are packed. metadata.json, text.json and *.npz stay as plain files next to
data.mdb, which is where the UniTEX-FLUX loader looks for metadata.json. With --out, the
envs are written to a mirror tree and the plain files (plus training_uid.json and caption/)
are copied there, so --out is a complete dataset root.

Needs only the `lmdb` package (uv sync --extra lmdb); no bpy.
"""

import argparse
import math
import os
import pathlib
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import lmdb


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
SUBSETS = ("render", "render_random")
PAGE = 4096


def image_files(d):
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def map_size_for(files):
    # Values above a page are stored in overflow pages: round each one up, then leave
    # headroom for the B-tree and free pages. LMDB grows a reader's map to the used size.
    pages = sum(math.ceil((f.stat().st_size + 64) / PAGE) + 1 for f in files)
    return int(pages * PAGE * 1.5) + (8 << 20)


def env_matches(dst, files):
    """True when dst already holds exactly these keys with the same byte sizes."""
    if not (dst / "data.mdb").exists():
        return False
    try:
        env = lmdb.open(str(dst), readonly=True, lock=False)
    except lmdb.Error:
        return False
    try:
        with env.begin() as txn:
            if txn.stat()["entries"] != len(files):
                return False
            for f in files:
                v = txn.get(f.stem.encode())
                if v is None or len(v) != f.stat().st_size:
                    return False
        return True
    finally:
        env.close()


def pack_dir(task):
    """Pack one directory. Returns (src, n_files, n_bytes, status)."""
    src, dst, delete_png, overwrite = task
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    files = image_files(src)
    if not files:
        if dst != src and (src / "data.mdb").exists():
            # Already packed in place with --delete-png: mirror the env and the plain files,
            # else --out yields uids in training_uid.json with no data behind them.
            dst.mkdir(parents=True, exist_ok=True)
            for p in src.iterdir():
                if p.is_file() and p.name != "lock.mdb":
                    shutil.copy2(p, dst / p.name)
            return str(src), 0, 0, "copied packed env"
        return str(src), 0, 0, "no images"
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        return str(src), len(files), 0, "error: two images share a basename"
    dst.mkdir(parents=True, exist_ok=True)
    n_bytes = sum(f.stat().st_size for f in files)
    if not overwrite and env_matches(dst, files):
        status = "exists"
    else:
        for name in ("data.mdb", "lock.mdb"):
            if (dst / name).exists():
                (dst / name).unlink()
        env = lmdb.open(str(dst), map_size=map_size_for(files), subdir=True, meminit=False)
        with env.begin(write=True) as txn:
            for f in files:
                txn.put(f.stem.encode(), f.read_bytes())
        env.sync()
        env.close()
        status = "packed"
    env = lmdb.open(str(dst), readonly=True, lock=False)
    with env.begin() as txn:
        bad = [f.name for f in files if txn.get(f.stem.encode()) != f.read_bytes()]
    env.close()
    if bad:
        return str(src), len(files), n_bytes, f"error: {len(bad)} values differ ({bad[0]})"
    if dst != src:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() not in IMAGE_EXTS and p.name not in ("data.mdb", "lock.mdb"):
                shutil.copy2(p, dst / p.name)
    if delete_png:
        for f in files:
            f.unlink()
        status += ", images deleted"
    return str(src), len(files), n_bytes, status


def find_dirs(root, subsets):
    dirs = []
    for subset in subsets:
        base = root / subset
        if not base.is_dir():
            continue
        for d in sorted(base.glob("*/*")):
            if d.is_dir():
                dirs.append(d)
    return dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", required=True, help="mvgen --out folder")
    parser.add_argument("--out", default=None, help="write envs to this mirror root instead of in place")
    parser.add_argument("--subsets", default=",".join(SUBSETS), help="comma-separated, default render,render_random")
    parser.add_argument("--delete-png", action="store_true", help="delete the images after a verified pack")
    parser.add_argument("--overwrite", action="store_true", help="re-pack even when a matching env exists")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else root
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    dirs = find_dirs(root, subsets)
    if args.limit:
        dirs = dirs[:args.limit]
    tasks = [(str(d), str(out / d.relative_to(root)), args.delete_png, args.overwrite) for d in dirs]
    print(f"pack_lmdb: {len(tasks)} directories under {root} -> {out}, {args.workers} workers")

    if out != root:
        out.mkdir(parents=True, exist_ok=True)
        if (root / "training_uid.json").exists():
            shutil.copy2(root / "training_uid.json", out / "training_uid.json")
        if (root / "caption").is_dir():
            shutil.copytree(root / "caption", out / "caption", dirs_exist_ok=True)

    t0, n_err, n_files, n_bytes = time.time(), 0, 0, 0
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(pack_dir, tasks, chunksize=4)
            results = list(results)
    else:
        results = [pack_dir(t) for t in tasks]
    for src, nf, nb, status in results:
        rel = pathlib.Path(src).relative_to(root)
        if status.startswith("error"):
            n_err += 1
            print(f"  [{rel}] {status}")
        n_files += nf
        n_bytes += nb
    print(f"pack_lmdb: {len(results) - n_err}/{len(results)} ok, {n_files} images, "
          f"{n_bytes / 2**20:.1f} MiB in {time.time() - t0:.1f} s")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
