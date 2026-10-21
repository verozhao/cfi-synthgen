"""
Open Food Facts records by UPC, for the back panel (back_panel.py): nutrition, serving size, ingredients,
allergens, brand. Public read-only API, only the code is sent. One <code>.json per product found (the 12-digit
UPC is tried as is and with a leading 0, as an EAN-13). The API answers 429 after a burst of requests, so lookups
are spaced and a 429 is retried with a growing wait. Records already in the folder are kept.

  python -m unitex.off_lookup --out OFF_DIR --skus a,b | sku_file
"""
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request

FIELDS = ("code,product_name,brands,quantity,serving_size,nutrition_data_per,nutriments,ingredients_text_en,"
          "ingredients_text,allergens_tags,categories_tags")
URL = "https://world.openfoodfacts.org/api/v2/product/{}.json?fields=" + FIELDS


def fetch(code, retries=4, wait=20.0):
    """The product record, None when the code is unknown. Raises after repeated 429s or other HTTP errors."""
    req = urllib.request.Request(URL.format(code), headers={"User-Agent": "cfi-texture-research/0.1"})
    for i in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.load(r)
            return d["product"] if d.get("status") == 1 else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code != 429 or i == retries:
                raise
            time.sleep(wait * (i + 1))
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--skus", required=True, help="file with one code per line, or a comma list")
    p.add_argument("--gap", type=float, default=4.0, help="seconds between requests")
    a = p.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = pathlib.Path(a.skus)
    codes = [l.split("#")[0].strip() for l in open(f)] if f.is_file() else a.skus.split(",")
    for code in [c for c in codes if c]:
        if (out / f"{code}.json").exists():
            print(f"{code}: kept")
            continue
        rec = None
        for c in (code, "0" + code):
            rec = fetch(c)
            time.sleep(a.gap)
            if rec is not None:
                break
        if rec is None:
            print(f"{code}: not in Open Food Facts")
            continue
        (out / f"{code}.json").write_text(json.dumps(rec, indent=1))
        n = rec.get("nutriments", {})
        print(f"{code}: {rec.get('product_name')!r}, serving {rec.get('serving_size')!r}, "
              f"kcal per serving {'energy-kcal_serving' in n}, ingredients {bool(rec.get('ingredients_text_en'))}")


if __name__ == "__main__":
    main()
