# Use Gemini to estimate a per-product longest-dimension and write it into
# each manifest_entry.json sidecar as target_size_m_estimated (meters).
# The file reads (title, shape) from manifest_entry.json, asks Gemini for the typical
# longest physical dimension of that product, validates the answer, writes back.
import argparse
import json
import os
import pathlib
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# Sanity bounds (meters). Gemini answers outside this are rejected.
MIN_SIZE_M = 0.03
MAX_SIZE_M = 1.20

PROMPT_TEMPLATE = """\
Product title: "{title}"
Shape category: {shape}

Estimate the longest physical dimension of this real-world retail product in centimeters.
For jars, bottles, cans, and box_rounded products this is usually the height.
For boxes, bags, and pouches it is whichever side is longest.
Use the quantity / weight in the title (e.g. "13 oz", "2.25 oz", "Family Size") to scale.

Reply with ONLY a single number (the centimeter value). No units, no explanation, no sentences."""


def parse_cm(raw: str) -> float | None:
    """Pull the first plausible number out of Gemini's reply."""
    if not raw:
        return None
    cleaned = raw.strip().split()[0].rstrip(".,;:").lstrip("~≈")
    try:
        return float(cleaned)
    except ValueError:
        # Maybe Gemini wrapped it: "Approximately 13.5 cm"
        import re
        m = re.search(r"(\d+\.?\d*)", raw)
        return float(m.group(1)) if m else None


def call_gemini(client, model, title, shape):
    from google.genai import types

    parts = [types.Part.from_text(text=PROMPT_TEMPLATE.format(title=title, shape=shape))]
    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(temperature=0, response_modalities=["TEXT"])

    last_err = None
    for attempt in range(4):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return (response.text or "").strip()
        except Exception as e:
            msg = str(e).lower()
            transient = "503" in msg or "429" in msg or "unavailable" in msg or "exhausted" in msg
            if not transient or attempt == 3:
                raise
            wait = 2 ** attempt * 5
            print(f"    transient ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
            last_err = e
    raise RuntimeError(f"Gemini call failed: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./approved_bundle")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--redo", action="store_true",
                    help="Overwrite target_size_m_estimated even if already set")
    ap.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"))
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: set GEMINI_API_KEY env var or pass --api-key", file=sys.stderr)
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=args.api_key)

    bundle = pathlib.Path(args.bundle)
    sidecars = sorted(bundle.rglob("manifest_entry.json"))
    if not sidecars:
        print(f"No manifest_entry.json under {bundle}")
        sys.exit(1)

    print(f"Found {len(sidecars)} sidecars")
    print(f"Model: {args.model}\n")

    n_done = n_skip = n_fail = 0
    for path in sidecars:
        entry = json.loads(path.read_text())
        sku = entry.get("sku") or path.parent.name
        title = entry.get("title")
        shape = entry.get("shape")

        if not title or not shape:
            print(f"  {sku:20s}  SKIP (missing title or shape)")
            n_skip += 1
            continue

        if "target_size_m_estimated" in entry and not args.redo:
            print(f"  {sku:20s}  SKIP (already set: {entry['target_size_m_estimated']:.3f}m)")
            n_skip += 1
            continue

        try:
            raw = call_gemini(client, args.model, title, shape)
        except Exception as e:
            print(f"  {sku:20s}  FAIL ({type(e).__name__}: {e})")
            n_fail += 1
            continue

        cm = parse_cm(raw)
        if cm is None:
            print(f"  {sku:20s}  FAIL (unparseable: {raw!r})")
            n_fail += 1
            continue

        size_m = cm / 100.0
        if not (MIN_SIZE_M <= size_m <= MAX_SIZE_M):
            print(f"  {sku:20s}  FAIL (out of range: {size_m:.3f}m from {raw!r})")
            n_fail += 1
            continue

        entry["target_size_m_estimated"] = round(size_m, 3)
        path.write_text(json.dumps(entry, indent=2))
        print(f"  {sku:20s}  {size_m:.3f}m  ({shape}, '{title[:50]}')")
        n_done += 1

    print(f"\nWrote: {n_done}, Skipped: {n_skip}, Failed: {n_fail}")


if __name__ == "__main__":
    main()
