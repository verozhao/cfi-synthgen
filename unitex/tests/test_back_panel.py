import numpy as np
import pytest

pytest.importorskip("cv2")

from unitex import back_panel as B  # noqa: E402


def test_upc_check_digit():
    assert B.upc_valid("016000233164")
    assert not B.upc_valid("016000233165")            # wrong check digit
    assert not B.upc_valid("01600023316")             # 11 digits
    assert not B.upc_valid("01600023316x")


def test_upc_bits_decode_back_to_the_code():
    code = "016000233164"
    bits = B.upc_bits(code)
    assert len(bits) == 95 and bits[:3] == "101" and bits[45:50] == "01010" and bits[-3:] == "101"
    left = [bits[3 + 7 * i:10 + 7 * i] for i in range(6)]
    right = [bits[50 + 7 * i:57 + 7 * i] for i in range(6)]
    assert all(c.count("1") % 2 == 1 for c in left)     # L codes have odd parity, R codes even
    assert all(c.count("1") % 2 == 0 for c in right)
    inv = {"".join("1" if b == "0" else "0" for b in c): str(d) for d, c in enumerate(B.L_CODE)}
    decoded = "".join(str(B.L_CODE.index(c)) for c in left) + "".join(inv[c] for c in right)
    assert decoded == code


def test_fda_rounding():
    assert [B.fda_round(v, "g", "fat") for v in (0.4, 2.26, 7.6)] == ["0", "2.5", "8"]
    assert [B.fda_round(v, "mg", "sodium") for v in (3, 137, 691)] == ["0", "135", "690"]
    assert [B.fda_round(v, "mg", "cholesterol") for v in (1, 3, 27)] == ["0", "<5", "25"]
    assert [B.fda_round(v, "g", "carbohydrates") for v in (0.3, 0.7, 30.4)] == ["0", "<1", "30"]
    assert [B.kcal_round(v) for v in (4, 47, 143)] == [0, 45, 140]


def test_serving_and_mass_strings():
    assert B.clean_serving("Amount per serving (318 g)") == "318 g"
    assert B.clean_serving("2 tbsp (33 g) (33 g)") == "2 tbsp (33 g)"
    assert B.clean_serving("1 portion (524.466 g)") == "1 portion (524 g)"
    assert B.clean_serving("1 cup (36 g)") == "1 cup (36 g)"
    assert B.clean_serving(None) is None
    assert B.grams("1 cup (36 g)") == 36.0
    assert abs(B.grams("18.6 oz") - 527.3) < 0.1
    assert B.grams(None) is None


def test_nutrition_rows_only_what_the_record_has():
    rec = {"serving_size": "1 cup (36 g)", "quantity": "18.6 oz (527 g)",
           "nutriments": {"energy-kcal_serving": 143, "fat_serving": 1.5, "sodium_serving": 0.2,
                          "carbohydrates_serving": 30.4, "energy-kcal_100g": 397}}
    nut = B.nutrition_rows(rec)
    assert nut["per_serving"] and nut["kcal"] == 140 and abs(nut["servings"] - 527 / 36) < 1e-6
    rows = {r[0]: r for r in nut["rows"]}
    assert set(rows) == {"Total Fat", "Sodium", "Total Carbohydrate"}   # no made-up rows
    assert rows["Total Fat"][1:3] == ("1.5g", "2%") and rows["Sodium"][1:3] == ("200mg", "9%")
    per100 = B.nutrition_rows({"nutriments": {"energy-kcal_100g": 360, "fat_100g": 0.0}})
    assert not per100["per_serving"] and per100["kcal"] == 360 and per100["rows"][0][2] == ""   # no %DV
    assert B.nutrition_rows({"nutriments": {"fat_100g": 1.0}}) is None    # no energy: no panel


def test_ingredients_and_allergens():
    ing, contains = B.ingredients_text({"ingredients_text_en": "sugar,  wheat flour, soy lecithin.",
                                        "allergens_tags": ["en:gluten", "en:soybeans", "en:unknown"]})
    assert ing == "SUGAR, WHEAT FLOUR, SOY LECITHIN."
    assert contains == "CONTAINS WHEAT, SOY INGREDIENTS."
    assert B.ingredients_text({"ingredients_text_en": "  "}) is None


def test_flat_base_is_the_dominant_colour_not_the_print():
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = (30, 160, 40)                              # green label
    img[10:30, 10:50] = 255                             # a large white made-up print box, 20 % of the area
    core = np.ones((64, 64), bool)
    base = B.base_fill(img, core, 64)
    assert np.allclose(base[0, 0], (30, 160, 40), atol=1) and np.allclose(base[20, 20], base[0, 0])


def test_blur_base_turns_white_print_boxes_into_label_colour():
    img = np.zeros((96, 96, 3), np.uint8)
    img[:] = (30, 160, 40)
    img[20:60, 20:70] = 255                             # a white box too large for the median filter
    base = B.base_fill(img, np.ones((96, 96), bool), 96, "blur", sigma_frac=1 / 16)
    assert np.allclose(base[40, 45], (30, 160, 40), atol=2)


def test_blur_base_keeps_a_white_label_and_its_border():
    img = np.full((96, 96, 3), 255, np.uint8)
    img[:12] = img[-12:] = (230, 30, 140)               # pink border, white label
    base = B.base_fill(img, np.ones((96, 96), bool), 96, "blur", sigma_frac=1 / 48)
    assert base[48, 48].min() > 240 and base[2, 48][1] < 100


def test_front_edges_cleaned_only_off_the_photo_where_the_surface_turns_away():
    R = 64
    view = np.zeros((R, R, 3), np.uint8)
    view[:] = (30, 160, 40)
    view[:, 52:58] = 255                                # made-up print near the right edge
    mask = np.zeros((R, R), bool)
    mask[:, 4:60] = True
    normals = np.zeros((R, R, 3))
    normals[mask] = B.PF.TO_FRONT_CAMERA
    normals[:, 48:60] = (0.9, -0.3, 0.0)                # cos 0.3 to the front camera: turning away
    w_photo = np.zeros((R, R), np.float32)
    w_photo[:32] = 1.0                                  # the photo was used on the top half only
    out = B.clean_front_edges(view, mask, normals, w_photo, R)
    assert np.allclose(out[48, 55], (30, 160, 40), atol=2)      # bottom: print replaced by label colour
    assert out[10, 55].min() == 255                             # top: photo area untouched
    assert np.array_equal(out[48, 20], view[48, 20])            # facing the camera: untouched


def test_inner_rect_stays_inside_the_region():
    region = np.zeros((100, 100), bool)
    region[20:80, 30:70] = True
    region[50, 69] = False                             # a notch
    x0, y0, x1, y1 = B.inner_rect(region)
    assert 30 <= x0 < x1 <= 70 and 20 <= y0 < y1 <= 80
    assert region[y0:y1, x0:x1].mean() >= 0.985
    assert B.inner_rect(np.zeros((100, 100), bool)) is None


def test_identity_colour_map():
    img = (np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5)
    M = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
    assert np.array_equal(B.apply_color_map(img, M), img)


def test_colour_map_on_the_blur_keeps_the_print_contrast():
    img = np.full((64, 64, 3), 200, np.uint8)
    img[30:34, 8:56] = 20                               # a dark text stroke on a light label
    M = np.concatenate([0.5 * np.eye(3), np.full((3, 1), 60.0)], axis=1)   # halves the contrast
    full = B.apply_color_map(img, M).astype(int)
    kept = B.apply_color_map(img, M, detail_sigma=4).astype(int)
    assert full[5, 5, 0] - full[32, 30, 0] == 90                   # contrast halved
    assert kept[5, 5, 0] - kept[32, 30, 0] > 130                   # stroke keeps most of its 180 contrast
    assert abs(kept[5, 5, 0] - full[5, 5, 0]) <= 3                 # the label colour is still mapped
