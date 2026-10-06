"""
Text-fidelity metrics in the style of GlyphAnchor's InfoTextBench (pure python).

  NED           1 - lev(gt, pred) / max(len(gt), len(pred)) over the reading-order
                concatenation of lines (lines joined by one space)
  word P / R/F1 multiset match of word tokens (case-folded, alphanumeric, >= 2 alnum chars)
  phrase hit    fraction of ground-truth lines found with line-level NED >= 0.8, where a
                line's score is the best of (a) NED against each predicted line and
                (b) NED against the best-matching window of the predicted concatenation
  line NED      that best score per ground-truth line

rapidfuzz is used for Levenshtein when installed, otherwise a pure python DP.

Normalization (normalize):
  - NFKC, then case-fold ("ß" -> "ss", full-width digits -> ASCII).
  - Curly quotes and dashes are mapped to their ASCII forms first.
  - Letters and digits of any script are kept, accents are kept unless fold_accents=True.
  - "." and "," between two digits are kept, so "10.75" and "1,000" stay one token.
  - "'" and "&" between two alphanumerics are dropped and the sides joined:
    "Campbell's" -> "campbells", "M&M" -> "mm".
  - Every other character (hyphen, slash, other punctuation, symbols) becomes a space:
    "Cheez-It" -> "cheez it", "NET WT/PESO" -> "net wt peso".
  - Runs of whitespace collapse to one space.

Tokenization (tokenize) additionally splits digit/letter boundaries ("12oz" -> "12 oz",
"18.5fl" -> "18.5 fl"), because packaging and OCR place the space inconsistently, and
then keeps tokens with at least 2 alphanumeric characters.
"""

import unicodedata
from collections import Counter

try:
    from rapidfuzz.distance import Levenshtein as _RF_LEV
except Exception:          # rapidfuzz is optional
    _RF_LEV = None

PHRASE_THRESHOLD = 0.8
HEIGHT_BUCKETS = (("<8", 0.0, 8.0), ("8-16", 8.0, 16.0), ("16-32", 16.0, 32.0), (">=32", 32.0, float("inf")))
BUCKET_NAMES = tuple(b[0] for b in HEIGHT_BUCKETS)

_CHAR_MAP = str.maketrans({       # typographic quotes and dashes -> ASCII (escapes keep the source ASCII)
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
})
_JOINERS = "'&"
_DIGIT_PUNCT = ".,"


# ────────────────────────────────────────────────────────────────────────────
# Normalization and tokens
# ────────────────────────────────────────────────────────────────────────────

def _fold_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(s, fold_accents=False):
    """Case-folded, punctuation-stripped form used by every metric (see module docstring)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).translate(_CHAR_MAP).casefold()
    if fold_accents:
        s = _fold_accents(s)
    out = []
    n = len(s)
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch)
            continue
        prev_c = s[i - 1] if i > 0 else ""
        next_c = s[i + 1] if i + 1 < n else ""
        if ch in _DIGIT_PUNCT and prev_c.isdigit() and next_c.isdigit():
            out.append(ch)
        elif ch in _JOINERS and prev_c.isalnum() and next_c.isalnum():
            continue
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _split_units(tok):
    """'12oz' -> ['12', 'oz'], '18.5fl' -> ['18.5', 'fl'], 'b12' -> ['b', '12']."""
    parts, cur, cur_digit = [], "", None
    for i, ch in enumerate(tok):
        if ch in _DIGIT_PUNCT:          # only survives normalize() between two digits
            cur += ch
            continue
        is_digit = ch.isdigit()
        if cur and cur_digit is not None and is_digit != cur_digit:
            parts.append(cur)
            cur = ""
        cur += ch
        cur_digit = is_digit
    if cur:
        parts.append(cur)
    return parts


def tokenize(s, fold_accents=False, split_units=True, min_alnum=2):
    """Word tokens of a string: normalized, unit-split, at least `min_alnum` alphanumerics."""
    toks = []
    for t in normalize(s, fold_accents).split():
        for p in (_split_units(t) if split_units else [t]):
            if sum(c.isalnum() for c in p) >= min_alnum:
                toks.append(p)
    return toks


def join_lines(lines, fold_accents=False):
    """Reading-order concatenation used for NED: normalized non-empty lines joined by one space."""
    return " ".join(x for x in (normalize(l, fold_accents) for l in lines) if x)


# ────────────────────────────────────────────────────────────────────────────
# Edit distance
# ────────────────────────────────────────────────────────────────────────────

def levenshtein(a, b):
    if _RF_LEV is not None:
        return _RF_LEV.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def ned(gt, pred):
    """1 - lev / max(len). Both empty -> 1.0."""
    d = max(len(gt), len(pred))
    return 1.0 if d == 0 else 1.0 - levenshtein(gt, pred) / d


def best_window_ned(pattern, text):
    """Best NED of `pattern` against any substring (window) of `text`.

    Sellers' approximate substring DP with the start of the best alignment tracked per cell,
    so the score uses max(len(pattern), len(window)) like ned().
    """
    m, n = len(pattern), len(text)
    if m == 0:
        return 1.0
    if n == 0:
        return 0.0
    prev = [0] * (n + 1)                 # empty pattern matches the empty window ending at j
    prev_start = list(range(n + 1))
    for i in range(1, m + 1):
        pc = pattern[i - 1]
        cur = [i] + [0] * n
        cur_start = [0] * (n + 1)
        for j in range(1, n + 1):
            best = prev[j - 1] + (pc != text[j - 1])
            start = prev_start[j - 1]
            v = prev[j] + 1              # pattern char unmatched
            if v < best:
                best, start = v, prev_start[j]
            v = cur[j - 1] + 1           # extra text char inside the window
            if v < best:
                best, start = v, cur_start[j - 1]
            cur[j] = best
            cur_start[j] = start
        prev, prev_start = cur, cur_start
    out = 0.0
    for j in range(n + 1):
        denom = max(m, j - prev_start[j])
        out = max(out, 1.0 - prev[j] / denom)
    return out


def line_score(gt_line, pred_lines, fold_accents=False):
    """Best line-level NED of one GT line against predicted lines or a window of their concat.

    Returns (score, source) where source is "line:<k>", "window" or None.
    """
    g = normalize(gt_line, fold_accents)
    if not g:
        return None, None
    preds = [normalize(p, fold_accents) for p in pred_lines]
    best, src = 0.0, None
    for k, p in enumerate(preds):
        if p:
            v = ned(g, p)
            if v > best:
                best, src = v, f"line:{k}"
    concat = " ".join(p for p in preds if p)
    if concat:
        v = best_window_ned(g, concat)
        if v > best + 1e-12:
            best, src = v, "window"
    return best, src


# ────────────────────────────────────────────────────────────────────────────
# Per-image metrics
# ────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b):
    return a / b if b else None


def prf(n_match, n_gt, n_pred):
    """Word precision / recall / F1. Precision is 0 (not undefined) when nothing was predicted
    but GT words exist, so an empty prediction cannot look precise. All None when n_gt == 0."""
    if n_gt == 0:
        return None, None, None
    r = n_match / n_gt
    p = n_match / n_pred if n_pred else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def compute_metrics(gt_lines, pred_lines, phrase_threshold=PHRASE_THRESHOLD, fold_accents=False):
    """All InfoTextBench-style metrics of one prediction against one GT (both lists of lines).

    Counts are returned alongside the ratios so results can be pooled exactly (aggregate()).
    `lines` has one entry per GT line (same order, empty-after-normalization lines included
    with ned None), carrying the line's best NED, hit flag and its word allocation.
    """
    gt_lines = list(gt_lines)
    pred_lines = [p for p in pred_lines if normalize(p, fold_accents)]
    pred_norm = {normalize(p, fold_accents) for p in pred_lines}
    gt_cat = join_lines(gt_lines, fold_accents)
    pred_cat = join_lines(pred_lines, fold_accents)
    lev = levenshtein(gt_cat, pred_cat)
    denom = max(len(gt_cat), len(pred_cat))

    gt_tok = [tokenize(g, fold_accents) for g in gt_lines]
    pred_counter = Counter(t for p in pred_lines for t in tokenize(p, fold_accents))
    gt_counter = Counter(t for ts in gt_tok for t in ts)
    n_gt_w = sum(gt_counter.values())
    n_pred_w = sum(pred_counter.values())
    n_match = sum(min(c, pred_counter[w]) for w, c in gt_counter.items())
    p, r, f = prf(n_match, n_gt_w, n_pred_w)

    remaining = Counter(pred_counter)
    lines = []
    n_hits = n_scored = 0
    for g, toks in zip(gt_lines, gt_tok):
        score, src = line_score(g, pred_lines, fold_accents)
        m = 0
        for t in toks:
            if remaining[t] > 0:
                remaining[t] -= 1
                m += 1
        hit = None if score is None else bool(score >= phrase_threshold)
        if score is not None:
            n_scored += 1
            n_hits += int(hit)
        lines.append({
            "ned": None if score is None else round(score, 4),
            "hit": hit,
            "exact": None if score is None else normalize(g, fold_accents) in pred_norm,
            "src": src,
            "n_words": len(toks),
            "n_words_matched": m,
        })

    missed = list((gt_counter - pred_counter).elements())
    extra = list((pred_counter - gt_counter).elements())
    return {
        "ned": None if denom == 0 else round(1 - lev / denom, 4),
        "lev": lev,
        "ned_denom": denom,
        "word_precision": None if p is None else round(p, 4),
        "word_recall": None if r is None else round(r, 4),
        "word_f1": None if f is None else round(f, 4),
        "n_gt_words": n_gt_w,
        "n_pred_words": n_pred_w,
        "n_match_words": n_match,
        "phrase_hit": None if n_scored == 0 else round(n_hits / n_scored, 4),
        "n_phrase_hits": n_hits,
        "n_gt_lines": n_scored,
        "line_ned": [l["ned"] for l in lines],
        "lines": lines,
        "missed_words": missed,
        "extra_words": extra,
        "gt_concat": gt_cat,
        "pred_concat": pred_cat,
    }


def height_bucket(h):
    """Name of the text-height bucket (px in the 512 px front view)."""
    if h is None:
        return None
    for name, lo, hi in HEIGHT_BUCKETS:
        if lo <= h < hi:
            return name
    return None


# ────────────────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────────────────

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(metrics_list):
    """Mean over images (images with no GT words are skipped per metric) and pooled totals."""
    ms = [m for m in metrics_list if m]
    mean = {k: _mean([m.get(k) for m in ms])
            for k in ("ned", "word_precision", "word_recall", "word_f1", "phrase_hit")}
    lev = sum(m["lev"] for m in ms)
    den = sum(m["ned_denom"] for m in ms)
    n_gt = sum(m["n_gt_words"] for m in ms)
    n_pred = sum(m["n_pred_words"] for m in ms)
    n_match = sum(m["n_match_words"] for m in ms)
    p, r, f = prf(n_match, n_gt, n_pred)
    hits = sum(m["n_phrase_hits"] for m in ms)
    nl = sum(m["n_gt_lines"] for m in ms)
    pooled = {
        "ned": None if den == 0 else round(1 - lev / den, 4),
        "word_precision": None if p is None else round(p, 4),
        "word_recall": None if r is None else round(r, 4),
        "word_f1": None if f is None else round(f, 4),
        "phrase_hit": None if nl == 0 else round(hits / nl, 4),
        "n_gt_words": n_gt, "n_pred_words": n_pred, "n_match_words": n_match,
        "n_phrase_hits": hits, "n_gt_lines": nl,
    }
    return {"n": len(ms), "mean": mean, "pooled": pooled}
