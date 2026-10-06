"""Unit tests for unitex.text_metrics (pure python, no OCR)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from unitex import text_metrics as tm


# ────────────────────────────────────────────────────────────────────────────
# Normalization and tokens
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("PURE LEAF.", "pure leaf"),
    ("Campbell's", "campbells"),
    ("Campbell’s", "campbells"),              # curly apostrophe
    ("M&M's", "mms"),
    ("Cheez-It", "cheez it"),
    ("NET WT/PESO NETO", "net wt peso neto"),
    ("10.75 OZ.", "10.75 oz"),
    ("1,000 mg", "1,000 mg"),
    ("(12 FL OZ)", "12 fl oz"),
    ("  lots   of\tspace ", "lots of space"),
    ("１２oz", "12oz"),                     # full-width digits -> ASCII (NFKC)
    ("STRASSE ß", "strasse ss"),               # case-fold
    ("JALAPEÑO", "jalapeño"),             # accents kept by default
    ("辛 라면", "辛 라면"),  # CJK / Hangul kept
    ("Vitalizes body and mind®.", "vitalizes body and mind"),
    ("", ""),
])
def test_normalize(raw, expected):
    assert tm.normalize(raw) == expected


def test_normalize_fold_accents():
    assert tm.normalize("CAFÉ INSTANTÁNEO", fold_accents=True) == "cafe instantaneo"


def test_tokenize_units_and_min_length():
    assert tm.tokenize("NET WT 12oz (340g)") == ["net", "wt", "12", "oz", "340"]
    assert tm.tokenize("18.5 FL.OZ") == ["18.5", "fl", "oz"]
    assert tm.tokenize("18.5fl oz") == ["18.5", "fl", "oz"]
    assert tm.tokenize("A 1 B2 x") == []                    # single alnum tokens dropped
    assert tm.tokenize("12oz", split_units=False) == ["12oz"]
    assert tm.tokenize("12 oz") == tm.tokenize("12oz")


# ────────────────────────────────────────────────────────────────────────────
# Edit distance
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("a, b, d", [
    ("", "", 0), ("abc", "", 3), ("", "ab", 2), ("kitten", "sitting", 3),
    ("flaw", "lawn", 2), ("same", "same", 0), ("ab", "ba", 2),
])
def test_levenshtein(a, b, d):
    assert tm.levenshtein(a, b) == d


def test_levenshtein_pure_python_fallback(monkeypatch):
    monkeypatch.setattr(tm, "_RF_LEV", None)
    assert tm.levenshtein("kitten", "sitting") == 3
    assert tm.levenshtein("", "abc") == 3
    assert tm.levenshtein("gumbo", "gambol") == 2


def test_ned():
    assert tm.ned("", "") == 1.0
    assert tm.ned("abc", "abc") == 1.0
    assert tm.ned("abc", "") == 0.0
    assert tm.ned("kitten", "sitting") == pytest.approx(1 - 3 / 7)


def test_best_window_ned():
    assert tm.best_window_ned("pure leaf", "unsweetened green tea pure leaf no sugar") == 1.0
    # one substitution inside a window
    assert tm.best_window_ned("pure leaf", "xx pure lcaf yy") == pytest.approx(1 - 1 / 9)
    # missing char: window "pureleaf" (8) vs pattern 9 -> 1 edit / 9
    assert tm.best_window_ned("pure leaf", "the pureleaf tea") == pytest.approx(1 - 1 / 9)
    assert tm.best_window_ned("abc", "") == 0.0
    assert tm.best_window_ned("", "abc") == 1.0
    assert tm.best_window_ned("abcdef", "xyz") == 0.0


def test_best_window_ned_matches_brute_force():
    import itertools
    import random
    rng = random.Random(0)
    alphabet = "abc "
    for _ in range(200):
        p = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
        t = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
        windows = [t[i:j] for i, j in itertools.combinations_with_replacement(range(len(t) + 1), 2)]
        brute = max(tm.ned(p, w) for w in windows)
        min_lev = min(tm.levenshtein(p, w) for w in windows)
        got = tm.best_window_ned(p, t)
        assert got <= brute + 1e-9                         # always the score of a real window
        assert got >= 1 - min_lev / len(p) - 1e-9          # at least the Sellers substring score


def test_line_score_sources():
    s, src = tm.line_score("PURE LEAF", ["NO SUGAR", "PURE LEAF"])
    assert s == 1.0 and src == "line:1"
    s, src = tm.line_score("GREEN TEA", ["UNSWEETENED GREEN", "TEA FLAVOR"])     # split across lines
    assert s == 1.0 and src == "window"
    s, src = tm.line_score("!!!", ["abc"])
    assert s is None and src is None


# ────────────────────────────────────────────────────────────────────────────
# Per-image metrics
# ────────────────────────────────────────────────────────────────────────────

def test_compute_metrics_perfect():
    gt = ["PURE LEAF", "NO SUGAR", "18.5 FL OZ"]
    m = tm.compute_metrics(gt, list(gt))
    assert m["ned"] == 1.0
    assert m["word_precision"] == m["word_recall"] == m["word_f1"] == 1.0
    assert m["phrase_hit"] == 1.0 and m["n_phrase_hits"] == 3
    assert m["missed_words"] == [] and m["extra_words"] == []
    assert all(l["exact"] for l in m["lines"])


def test_compute_metrics_multiset_and_order():
    gt = ["TEA TEA", "GREEN"]
    m = tm.compute_metrics(gt, ["GREEN TEA"])
    assert m["n_gt_words"] == 3 and m["n_pred_words"] == 2 and m["n_match_words"] == 2
    assert m["word_recall"] == pytest.approx(2 / 3, abs=1e-4)          # ratios are rounded to 4 places
    assert m["word_precision"] == 1.0
    assert m["missed_words"] == ["tea"]
    # NED is order sensitive: "tea tea green" vs "green tea"
    assert m["ned"] == pytest.approx(1 - tm.levenshtein("tea tea green", "green tea") / 13, abs=1e-4)
    # phrase: "GREEN" found in the window, "TEA TEA" not (best 0.43 < 0.8)
    assert m["lines"][1]["hit"] is True and m["lines"][0]["hit"] is False
    assert m["phrase_hit"] == 0.5


def test_compute_metrics_line_word_allocation():
    gt = ["RED BULL", "RED"]
    m = tm.compute_metrics(gt, ["RED BULL"])
    # the single predicted "red" is consumed by the first GT line
    assert [(l["n_words"], l["n_words_matched"]) for l in m["lines"]] == [(2, 2), (1, 0)]


def test_compute_metrics_empty_prediction():
    m = tm.compute_metrics(["PURE LEAF"], [])
    assert m["word_recall"] == 0.0 and m["word_precision"] == 0.0 and m["word_f1"] == 0.0
    assert m["ned"] == 0.0 and m["phrase_hit"] == 0.0
    assert m["lines"][0]["ned"] == 0.0


def test_compute_metrics_no_gt_words():
    m = tm.compute_metrics(["!!", "x"], ["anything"])
    assert m["word_recall"] is None and m["word_precision"] is None
    # "x" still normalizes to a non-empty line, "!!" does not
    assert m["n_gt_lines"] == 1 and m["lines"][0]["ned"] is None


def test_phrase_threshold_boundary():
    # 10-char line with 2 edits -> NED 0.8 counts as a hit (>=)
    m = tm.compute_metrics(["abcdefghij"], ["abcdefghXY"])
    assert m["lines"][0]["ned"] == pytest.approx(0.8)
    assert m["lines"][0]["hit"] is True
    m = tm.compute_metrics(["abcdefghij"], ["abcdefgXYZ"])
    assert m["lines"][0]["hit"] is False


def test_prf_edge_cases():
    assert tm.prf(0, 0, 5) == (None, None, None)
    assert tm.prf(0, 3, 0) == (0.0, 0.0, 0.0)
    p, r, f = tm.prf(2, 4, 2)
    assert (p, r) == (1.0, 0.5) and f == pytest.approx(2 / 3)


# ────────────────────────────────────────────────────────────────────────────
# Buckets and aggregation
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("h, b", [(0.0, "<8"), (7.99, "<8"), (8.0, "8-16"), (15.9, "8-16"),
                                  (16.0, "16-32"), (32.0, ">=32"), (400.0, ">=32"), (None, None)])
def test_height_bucket(h, b):
    assert tm.height_bucket(h) == b


def test_aggregate_mean_vs_pooled():
    a = tm.compute_metrics(["one two three four"], ["one two three four"])      # 4/4
    b = tm.compute_metrics(["five six"], [])                                    # 0/2
    c = tm.compute_metrics(["!!"], ["x"])                                       # no GT words, skipped in means
    agg = tm.aggregate([a, b, c])
    assert agg["n"] == 3
    assert agg["mean"]["word_recall"] == pytest.approx(0.5)
    assert agg["pooled"]["word_recall"] == pytest.approx(4 / 6, abs=1e-4)
    assert agg["pooled"]["n_gt_words"] == 6
    assert agg["pooled"]["phrase_hit"] == pytest.approx(1 / 2)
    lev = a["lev"] + b["lev"] + c["lev"]
    den = a["ned_denom"] + b["ned_denom"] + c["ned_denom"]
    assert agg["pooled"]["ned"] == pytest.approx(round(1 - lev / den, 4))


def test_aggregate_empty():
    agg = tm.aggregate([])
    assert agg["n"] == 0 and agg["mean"]["ned"] is None and agg["pooled"]["word_recall"] is None
