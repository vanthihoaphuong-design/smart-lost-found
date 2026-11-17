import os
import io
import uuid
import json
import re
import gc
import math
import colorsys
import threading
import urllib3
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func, text
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader

from PIL import Image, ImageOps, ImageFile

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

VN_TZ = timezone(timedelta(hours=7))

HAS_NUMPY = False
try:
    import numpy as np
    HAS_NUMPY = True
except Exception as e:
    print(f"Cảnh báo NumPy không khả dụng: {e}")

cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", "")
api_key = os.environ.get("CLOUDINARY_API_KEY", "")
api_secret = os.environ.get("CLOUDINARY_API_SECRET", "")

if cloud_name and api_key and api_secret:
    try:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True
        )
    except Exception as e:
        print(f"Lỗi config Cloudinary: {e}")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smart_lost_found_secret_key_123")


@app.template_filter("vn_date")
def format_vn_date(date_val):
    """Chuyển đổi chuỗi ngày YYYY-MM-DD hoặc YYYY/MM/DD sang DD-MM-YYYY."""
    if not date_val:
        return ""
    s = str(date_val).strip()
    if s.lower() in {"none", "null", ""}:
        return ""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s[:10], fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return s


UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 12_000_000

db_url = os.environ.get("DATABASE_URL", "sqlite:///lost_found.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_timeout": 30
}

try:
    from database import db
    db.init_app(app)
except Exception as e:
    print(f"Lỗi Init DB: {e}")

try:
    from models.lost_item import LostItem
    from models.found_item import FoundItem
except Exception as e:
    print(f"Lỗi Import Models: {e}")

with app.app_context():
    try:
        db.create_all()
        db.session.execute(text("ALTER TABLE found_item ADD COLUMN noi_luu_giu VARCHAR(200);"))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Thông báo DB migration: {e}")

CACHE_FILE = "feature_cache_v39.json"
CACHE_LOCK = threading.Lock()
MAX_CACHE_SIZE = 400


def _load_disk_cache():
    try:
        if os.path.isfile(CACHE_FILE):
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


FEATURE_CACHE = _load_disk_cache()


def _persist_disk_cache():
    try:
        with CACHE_LOCK:
            serializable = {
                k: v for k, v in FEATURE_CACHE.items()
                if isinstance(v, dict) and v.get("version") == 39
            }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)
    except Exception:
        pass


HTTP_SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=6, pool_maxsize=6, max_retries=1)
HTTP_SESSION.mount("http://", adapter)
HTTP_SESSION.mount("https://", adapter)
HTTP_SESSION.headers.update({"User-Agent": "SmartLostFound/1.0"})

STOP_WORDS_VN = {
    "con", "cái", "chiếc", "hộp", "bộ", "cặp", "đôi", "quả", "trái",
    "tấm", "bức", "tờ", "cục", "quyển", "cuốn", "cây", "vẽ", "màu", "loại", "để", "bàn", "học"
}
COLOR_WORDS_VN = ["đen", "đỏ", "xanh", "vàng", "trắng", "hồng", "tím", "nâu", "xám", "cam", "bạc"]


def normalize_location_text(loc):
    """Chuẩn hóa tên địa điểm (sửa Căng tin -> Căn tin)."""
    if not loc:
        return ""
    s = str(loc).strip()
    if s.lower() in {"none", "null", ""}:
        return ""
    return re.sub(r'căng\s*tin', 'Căn tin', s, flags=re.IGNORECASE)


def extract_colors(text):
    if not text:
        return set()
    t = str(text).lower()
    found = {c for c in COLOR_WORDS_VN if re.search(rf'\b{c}\b', t)}
    if "bạc" in found or "trắng" in found:
        found.update({"trắng", "bạc"})
    return found


def check_real_image(img_path):
    """Kiểm tra URL hoặc file ảnh tồn tại trước khi trích xuất đặc trưng."""
    if not img_path:
        return False
    try:
        value = str(img_path).strip()
        low = value.lower()
        if low in {"", "none", "null", "undefined"}:
            return False
        invalid = ["chua co anh", "chưa có ảnh", "no_image", "placeholder", "120?text="]
        if any(token in low for token in invalid):
            return False
        if low.startswith(("http://", "https://")):
            return True
        path = value if os.path.isabs(value) else os.path.join(
            app.config.get("UPLOAD_FOLDER", "static/uploads"), value
        )
        return os.path.isfile(path)
    except Exception:
        return False


def get_feature_cache(img_path):
    return FEATURE_CACHE.get(str(img_path)) if img_path else None


def save_feature_cache(img_path, features):
    if not img_path:
        return
    key = str(img_path)
    with CACHE_LOCK:
        if len(FEATURE_CACHE) >= MAX_CACHE_SIZE and key not in FEATURE_CACHE:
            FEATURE_CACHE.pop(next(iter(FEATURE_CACHE)))
        FEATURE_CACHE[key] = features


def _read_image_bytes(img_path, max_bytes=1536 * 1024):
    """Tải ảnh thu nhỏ siêu nhẹ (w_160) từ Cloudinary qua Session Keep-Alive."""
    value = str(img_path).strip()
    if "res.cloudinary.com/" in value and "/image/upload/" in value and "/w_" not in value:
        value = value.replace("/image/upload/", "/image/upload/w_160,c_limit,q_auto:low,f_jpg/", 1)
    if value.startswith(("http://", "https://")):
        response = HTTP_SESSION.get(value, timeout=(2.0, 4.0), verify=False)
        response.raise_for_status()
        if len(response.content) > max_bytes:
            return None
        return response.content

    path = value if os.path.isabs(value) else os.path.join(
        app.config.get("UPLOAD_FOLDER", "static/uploads"), value
    )
    if not os.path.isfile(path) or os.path.getsize(path) > max_bytes:
        return None
    with open(path, "rb") as file:
        return file.read(max_bytes + 1)


def _extract_foreground_pil(pil_img):
    """Tách nền nhẹ RAM trên lưới 48x48, cắt sát khung bao quanh vật thể."""
    size = 48
    small = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    px = list(small.getdata())

    border_samples = []
    for i in range(size):
        border_samples.append(px[i])
        border_samples.append(px[(size - 1) * size + i])
        border_samples.append(px[i * size])
        border_samples.append(px[i * size + (size - 1)])

    n_b = max(1, len(border_samples))
    mean_r = sum(p[0] for p in border_samples) / n_b
    mean_g = sum(p[1] for p in border_samples) / n_b
    mean_b = sum(p[2] for p in border_samples) / n_b
    _, bg_sat, _ = colorsys.rgb_to_hsv(mean_r / 255.0, mean_g / 255.0, mean_b / 255.0)

    bg_palette = border_samples[::4]

    mask = [False] * (size * size)
    min_x, min_y, max_x, max_y = size, size, 0, 0
    fg_count = 0

    for y in range(2, size - 2):
        row_off = y * size
        for x in range(2, size - 2):
            idx = row_off + x
            r, g, b = px[idx]
            _, sat, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

            min_bg_dist = 999999
            for br, bg_c, bb in bg_palette:
                dr = r - br
                dg = g - bg_c
                db = b - bb
                dist_sq = dr * dr + dg * dg + db * db
                if dist_sq < min_bg_dist:
                    min_bg_dist = dist_sq
                    if min_bg_dist < 500:
                        break

            sat_diff = abs(sat - bg_sat)
            if min_bg_dist > 800 or (sat_diff > 0.14 and min_bg_dist > 320):
                mask[idx] = True
                fg_count += 1
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y

    if fg_count < 12 or max_x <= min_x + 1 or max_y <= min_y + 1:
        w, h = pil_img.size
        cropped = pil_img.crop((int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82)))
        fallback_mask = [
            (9 <= (i % size) < 39 and 9 <= (i // size) < 39)
            for i in range(size * size)
        ]
        return cropped, small, fallback_mask

    w, h = pil_img.size
    pad_x = max(1, int((max_x - min_x) * 0.04))
    pad_y = max(1, int((max_y - min_y) * 0.04))
    x0 = max(0, int(((min_x - pad_x) / size) * w))
    y0 = max(0, int(((min_y - pad_y) / size) * h))
    x1 = min(w, int(((max_x + pad_x) / size) * w))
    y1 = min(h, int(((max_y + pad_y) / size) * h))

    cropped = pil_img.crop((x0, y0, max(x0 + 4, x1), max(y0 + 4, y1)))
    return cropped, small, mask


def _detect_visual_colors_masked(small_48, fg_mask):
    """Nhận diện màu sắc chủ đạo của vật thể."""
    try:
        px = list(small_48.getdata())
        counts = {}
        total = 0
        for i, m in enumerate(fg_mask):
            if not m:
                continue
            r, g, b = px[i]
            hue, sat, val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            if val > 0.94 and sat < 0.14:
                continue
            total += 1
            if val < 0.22:
                c_name = "đen"
            elif sat < 0.18 and val > 0.78:
                c_name = "trắng"
            elif sat < 0.22:
                c_name = "xám"
            else:
                deg = hue * 360.0
                if deg < 18 or deg >= 340:
                    c_name = "đỏ"
                elif deg < 42:
                    c_name = "cam"
                elif deg < 68:
                    c_name = "vàng"
                elif deg < 255:
                    c_name = "xanh"
                elif deg < 310:
                    c_name = "tím"
                else:
                    c_name = "hồng"
            counts[c_name] = counts.get(c_name, 0) + 1

        if total == 0:
            return set()
        detected = {c for c, cnt in counts.items() if (cnt / total) >= 0.18}
        if "trắng" in detected or "xám" in detected:
            detected.add("bạc")
        return detected
    except Exception:
        return set()


def _compute_masked_color_hist(small_48, fg_mask):
    """Histogram sắc độ bất biến ánh sáng trên đúng các điểm ảnh vật thể."""
    eq_48 = ImageOps.autocontrast(small_48, cutoff=2)
    px_eq = list(eq_48.getdata())
    fg_indices = [i for i, m in enumerate(fg_mask) if m] or list(range(48 * 48))

    bins = [0.0] * 28
    count = 0.0
    for idx_px in fg_indices:
        rv, gv, bv = px_eq[idx_px]
        h, s, v = colorsys.rgb_to_hsv(rv / 255.0, gv / 255.0, bv / 255.0)
        if v > 0.92 and s < 0.16:
            continue
        if s < 0.16:
            bins[min(3, int(v * 4.0))] += 1.0
        else:
            h_pos = (h * 12.0) % 12.0
            h0 = int(h_pos)
            frac = h_pos - h0
            bins[4 + h0] += (1.0 - frac * 0.45)
            bins[4 + ((h0 + 1) % 12)] += (frac * 0.45)
        rgb_sum = float(rv + gv + bv)
        if rgb_sum > 30.0:
            r_bin = min(3, int((rv / rgb_sum) * 4.0))
            g_bin = min(2, int((gv / rgb_sum) * 3.0))
            bins[16 + r_bin * 3 + g_bin] += 0.8
        count += 1.0

    if count <= 0.0:
        return [1.0 / 28.0] * 28
    inv = 1.0 / count
    return [x * inv for x in bins]


def _compute_fine_structure_8x8(fg_crop):
    """
    Trích xuất đặc trưng cấu trúc chi tiết lưới 8x8 (64 vùng) + Biểu đồ hướng cạnh HOG:
    Phân biệt rõ mặt dây chuyền hoa vs giọt nước, áo khoác trơn vs áo kẻ ngang, chuột trơn vs chuột LED.
    """
    base_64 = ImageOps.equalize(fg_crop.convert("L")).resize((64, 64), Image.Resampling.BILINEAR)
    rot_signatures = []

    for angle in (0, 90, 180, 270):
        g_img = base_64.rotate(angle) if angle else base_64
        g_px = list(g_img.getdata())

        # 1. Lưới 8x8 ô (mỗi ô 8x8 pixel): lưu độ sáng trung bình và mật độ biên ngang/dọc
        cell_vec = []
        hog_bins = [0.0] * 8

        for gy in range(8):
            for gx in range(8):
                lum_sum = 0.0
                dx_sum = 0.0
                dy_sum = 0.0
                cnt = 0
                for y in range(gy * 8, (gy + 1) * 8):
                    row_off = y * 64
                    for x in range(gx * 8, (gx + 1) * 8):
                        val = float(g_px[row_off + x])
                        lum_sum += val
                        if 0 < x < 63 and 0 < y < 63:
                            gx_val = float(g_px[row_off + x + 1]) - float(g_px[row_off + x - 1])
                            gy_val = float(g_px[row_off + 64 + x]) - float(g_px[row_off - 64 + x])
                            mag = math.hypot(gx_val, gy_val)
                            dx_sum += abs(gx_val)
                            dy_sum += abs(gy_val)
                            if mag > 18.0:
                                ang = (math.degrees(math.atan2(gy_val, gx_val)) + 180.0) % 180.0
                                b_idx = min(7, int((ang / 180.0) * 8.0))
                                hog_bins[b_idx] += mag
                        cnt += 1

                cell_vec.extend([
                    (lum_sum / max(1, cnt)) / 255.0,
                    (dx_sum / max(1, cnt)) / 128.0,
                    (dy_sum / max(1, cnt)) / 128.0,
                ])

        hog_norm = math.sqrt(sum(v * v for v in hog_bins)) or 1.0
        hog_vec = [v / hog_norm for v in hog_bins]

        rot_signatures.append({
            "cells": cell_vec,
            "hog": hog_vec
        })

    return rot_signatures


def calculate_image_features(img_path):
    """Trích xuất đặc trưng v39 siêu nhẹ RAM (< 8KB/ảnh)."""
    if not check_real_image(img_path):
        return None

    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        data = _read_image_bytes(img_path, max_bytes=1536 * 1024)
        if not data:
            return None

        with Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width < 2 or height < 2 or width * height > 12_000_000:
                return None

            image = ImageOps.exif_transpose(source).convert("RGB")
            fg_crop, small_48, fg_mask = _extract_foreground_pil(image)
            visual_colors = _detect_visual_colors_masked(small_48, fg_mask)
            fg_color_hist = _compute_masked_color_hist(small_48, fg_mask)
            rot_sigs = _compute_fine_structure_8x8(fg_crop)

            cw, ch = fg_crop.size
            aspect_ratio = min(cw, ch) / max(1.0, max(cw, ch))

            features = {
                "version": 39,
                "fg_color_hist": fg_color_hist,
                "rot_sigs": rot_sigs,
                "aspect_ratio": aspect_ratio,
                "visual_colors": list(visual_colors),
            }

        save_feature_cache(img_path, features)
        return features
    except Exception as exc:
        print(f"Lỗi trích xuất đặc trưng ảnh: {exc}")
        return None


def _cosine_sim(v1, v2):
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    if HAS_NUMPY:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(y * y for y in v2))
    return (dot / (n1 * n2)) if (n1 > 1e-6 and n2 > 1e-6) else 0.0


def _pearson_corr(v1, v2):
    """Tương quan Pearson chuẩn hóa trừ trung bình (nhạy với hoa văn cục bộ từng ô 8x8)."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    if HAS_NUMPY:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-6 else 0.0
    m1 = sum(v1) / len(v1)
    m2 = sum(v2) / len(v2)
    a = [x - m1 for x in v1]
    b = [y - m2 for y in v2]
    denom = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return (sum(x * y for x, y in zip(a, b)) / denom) if denom > 1e-6 else 0.0


def image_score(
    img1_filename,
    img2_filename,
    is_same_keyword=False,
    color_conflict=False
):
    """
    Chấm điểm thị giác v39 (0–50):
    - 16đ Sắc độ chuẩn hóa vật thể (HSV + Chromaticity).
    - 24đ Tương quan chi tiết lưới 8x8 (trừ trung bình để bắt khác biệt mặt dây chuyền, sọc áo, LED chuột).
    - 10đ Biểu đồ hướng cạnh HOG + tỷ lệ khung hình.
    """
    try:
        if not check_real_image(img1_filename) or not check_real_image(img2_filename):
            return 0, 0.0

        u1 = str(img1_filename).strip()
        u2 = str(img2_filename).strip()
        if u1 == u2:
            return 50, 1.0

        if color_conflict:
            return 10, 0.20

        f1 = calculate_image_features(u1)
        f2 = calculate_image_features(u2)
        if f1 is None or f2 is None:
            return 0, 0.0

        # 1. Điểm sắc độ vật thể (0–16đ)
        h1 = f1.get("fg_color_hist", [])
        h2 = f2.get("fg_color_hist", [])
        color_inter = sum(min(a, b) for a, b in zip(h1, h2)) if (h1 and h2) else 0.0
        color_pts = max(0, min(16, int(round(((color_inter - 0.14) / 0.36) * 16))))

        # 2. Điểm cấu trúc chi tiết lưới 8x8 + hướng cạnh HOG qua 4 góc xoay (0–32đ)
        s1_list = f1.get("rot_sigs", [])
        s2_list = f2.get("rot_sigs", [])
        if s1_list and s2_list:
            base_s1 = s1_list[0]
            best_struct = -1.0
            for s2 in s2_list:
                cell_corr = max(0.0, _pearson_corr(base_s1["cells"], s2["cells"]))
                hog_sim = max(0.0, _cosine_sim(base_s1["hog"], s2["hog"]))
                combined = 0.72 * cell_corr + 0.28 * hog_sim
                if combined > best_struct:
                    best_struct = combined

            # Ngưỡng tương quan chi tiết: 0.45 -> 4đ | 0.65 -> 16đ | >= 0.85 -> 32đ
            struct_pts = max(2, min(32, int(round(((best_struct - 0.42) / 0.44) * 32))))
        else:
            struct_pts = 14

        ar1 = float(f1.get("aspect_ratio", 0.7))
        ar2 = float(f2.get("aspect_ratio", 0.7))
        ar_pts = 2 if abs(ar1 - ar2) < 0.22 else 0

        total = max(10, min(48, color_pts + struct_pts + ar_pts))

        if not is_same_keyword and total > 12:
            total = 12

        return int(total), float(total / 50.0)
    except Exception as exc:
        print(f"Lỗi tính điểm hình ảnh: {exc}")
        return 0, 0.0


def _tokenize_meaningful_words(text_name):
    """Tách tập từ khóa có nghĩa tự nhiên (bỏ từ dừng & từ chỉ màu sắc)."""
    if not text_name:
        return set()
    t = re.sub(r'[^\w\s]', ' ', str(text_name).lower())
    words = [w for w in t.split() if len(w) >= 2 and w not in STOP_WORDS_VN and w not in COLOR_WORDS_VN]
    return set(words)


def get_core_keyword(text_name):
    """Chuẩn hóa danh từ cốt lõi, phân biệt cụm từ ghép (chuột máy tính != máy tính)."""
    if not text_name:
        return ""
    t = str(text_name).lower().strip()

    if "chuột" in t or "mouse" in t:
        return "chuột"

    synonyms_map = {
        "máy tính casio": ["casio", "fx580", "fx 580", "fx570", "fx 570", "880btg", "fx880", "fx 880", "máy tính bỏ túi"],
        "laptop": ["laptop", "macbook", "máy tính xách tay"],
        "điện thoại": ["điện thoại", "phone", "iphone", "samsung", "oppo", "xiaomi"],
        "balo": ["balo", "ba lô"],
        "ví": ["ví", "bóp"],
        "nón": ["nón", "mũ"],
        "thẻ": ["thẻ", "căn cước", "cccd"],
    }

    for core_key, words in synonyms_map.items():
        if any(w in t for w in words):
            nums = re.findall(r'\d+', t)
            return f"{core_key}_{'_'.join(nums)}" if nums else core_key

    tokens = sorted(_tokenize_meaningful_words(t))
    if not tokens:
        return t
    nums = re.findall(r'\d+', t)
    base = tokens[0]
    return f"{base}_{'_'.join(nums)}" if nums else base


def similarity(text1, text2):
    """So khớp tên tổng quát."""
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()

    if t1 == t2:
        return 1.0

    nums1 = re.findall(r'\d+', t1)
    nums2 = re.findall(r'\d+', t2)
    if nums1 and nums2 and nums1 != nums2:
        return 0.0

    if ("chuột" in t1) != ("chuột" in t2):
        return 0.0

    k1 = get_core_keyword(t1)
    k2 = get_core_keyword(t2)
    if k1 and k2 and k1 == k2:
        return 0.95

    w1 = _tokenize_meaningful_words(t1)
    w2 = _tokenize_meaningful_words(t2)
    common = w1.intersection(w2)
    if common:
        non_generic = {w for w in common if w not in {"sách", "vở", "tập", "đồ", "vật", "máy"}}
        if non_generic:
            overlap_ratio = len(non_generic) / max(1, min(len(w1), len(w2)))
            return min(0.95, 0.75 + 0.20 * overlap_ratio)
        return 0.0

    if len(t1) >= 3 and len(t2) >= 3 and (t1 in t2 or t2 in t1):
        return 0.85

    return 0.0


def evaluate_color_match(lost_item, found_item, is_same_key, visual_colors1=None, visual_colors2=None):
    """Tính điểm màu sắc (0–15), tự bổ sung màu chủ đạo từ HSV ảnh nếu bài đăng thiếu màu chữ."""
    m1 = str(lost_item.mau_sac or "").strip().lower()
    m2 = str(found_item.mau_sac or "").strip().lower()
    has_m1 = m1 not in ["none", "null", ""]
    has_m2 = m2 not in ["none", "null", ""]

    colors1 = extract_colors(m1) | extract_colors(lost_item.ten_do)
    colors2 = extract_colors(m2) | extract_colors(found_item.ten_do)

    if not colors1 and visual_colors1:
        colors1 = set(visual_colors1)
    if not colors2 and visual_colors2:
        colors2 = set(visual_colors2)

    if colors1 and colors2:
        if colors1.intersection(colors2):
            return 15, False
        else:
            return 0, True

    if has_m1 and has_m2:
        sim = SequenceMatcher(None, m1, m2).ratio()
        return int(sim * 15), (sim < 0.3)

    return (10 if is_same_key else 0), False


def location_score(loc1, loc2):
    """Chấm điểm vị trí (0–10)."""
    v1 = normalize_location_text(loc1).lower()
    v2 = normalize_location_text(loc2).lower()
    if not v1 or not v2:
        return 0
    if v1 == v2:
        return 10

    base1 = re.sub(r'\d+', '', v1).strip()
    base2 = re.sub(r'\d+', '', v2).strip()
    if base1 and base1 == base2:
        return 7

    if len(v1) >= 2 and len(v2) >= 2 and v1[:2] == v2[:2] and v1[0].isdigit():
        return 6

    sim = SequenceMatcher(None, v1, v2).ratio()
    if sim > 0.75:
        return 7
    elif sim > 0.45:
        return 4
    return 0


def parse_date_flexible(date_str):
    if not date_str or str(date_str).strip().lower() in ["none", "null", ""]:
        return None
    d_str = str(date_str).strip()
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]:
        try:
            return datetime.strptime(d_str, fmt)
        except Exception:
            pass
    return None


def date_score(date1, date2):
    d1, d2 = parse_date_flexible(date1), parse_date_flexible(date2)
    if not d1 or not d2:
        return 0
    diff = abs((d1 - d2).days)
    if diff == 0:
        return 10
    elif diff <= 5:
        return 9
    elif diff <= 15:
        return 7
    elif diff <= 30:
        return 4
    return 0


def get_current_pin():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return str(data.get("admin_pin", "123456"))
    except Exception:
        return "123456"


def save_new_pin(new_pin):
    try:
        config_data = {}
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            config_data = {}

        config_data["admin_pin"] = str(new_pin)

        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return False


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/lost", methods=["GET", "POST"])
def lost():
    if request.method == "POST":
        filename = ""
        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                filename = res.get("secure_url", "")
            except Exception:
                filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        if filename:
            calculate_image_features(filename)
            _persist_disk_cache()

        try:
            item = LostItem(
                ten_do=request.form.get("ten_do", ""),
                loai_do=request.form.get("loai_do", ""),
                mau_sac=request.form.get("mau_sac", ""),
                noi_mat=normalize_location_text(request.form.get("noi_mat", "")),
                ngay_mat=request.form.get("ngay_mat", ""),
                hinh_anh=filename,
                trang_thai="DANG_TIM"
            )
            db.session.add(item)
            db.session.commit()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
        return redirect(url_for("list_items"))
    return render_template("lost_form.html")


@app.route("/found", methods=["GET", "POST"])
def found():
    if request.method == "POST":
        filename = ""
        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                filename = res.get("secure_url", "")
            except Exception:
                filename = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        if filename:
            calculate_image_features(filename)
            _persist_disk_cache()

        try:
            item = FoundItem(
                ten_do=request.form.get("ten_do", ""),
                loai_do=request.form.get("loai_do", ""),
                mau_sac=request.form.get("mau_sac", ""),
                noi_nhat=normalize_location_text(request.form.get("noi_nhat", "")),
                ngay_nhat=request.form.get("ngay_nhat", ""),
                noi_luu_giu=normalize_location_text(request.form.get("noi_luu_giu", "")),
                hinh_anh=filename,
                trang_thai="DANG_LUU_GIU"
            )
            db.session.add(item)
            db.session.commit()
        except Exception:
            try:
                db.session.rollback()
            except Exception:
                pass
        return redirect(url_for("found_list"))
    return render_template("found_form.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    if filename.startswith("http://") or filename.startswith("https://"):
        return redirect(filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/lost-list")
def list_items():
    try:
        items = LostItem.query.order_by(LostItem.id.desc()).all()
        for item in items:
            item.noi_mat = normalize_location_text(item.noi_mat)
    except Exception:
        items = []
    return render_template("list_items.html", items=items)


@app.route("/found-list")
def found_list():
    try:
        items = FoundItem.query.order_by(FoundItem.id.desc()).all()
        for item in items:
            item.noi_nhat = normalize_location_text(item.noi_nhat)
            item.noi_luu_giu = normalize_location_text(item.noi_luu_giu)
    except Exception:
        items = []
    return render_template("found_list.html", items=items)


@app.route("/search")
def search():
    keyword = request.args.get("keyword", "").strip()
    category = request.args.get("category", "").strip()
    location = normalize_location_text(request.args.get("location", ""))
    color = request.args.get("color", "").strip()
    status = request.args.get("status", "").strip()

    try:
        lost_query = LostItem.query
        if keyword:
            lost_query = lost_query.filter(LostItem.ten_do.ilike(f"%{keyword}%"))
        if category:
            lost_query = lost_query.filter(LostItem.loai_do == category)
        if location:
            lost_query = lost_query.filter(LostItem.noi_mat.ilike(f"%{location}%"))
        if color:
            lost_query = lost_query.filter(LostItem.mau_sac.ilike(f"%{color}%"))

        if status in ("DANG_TIM", "DA_TIM"):
            lost_query = lost_query.filter(LostItem.trang_thai == status)
        elif status in ("DANG_LUU_GIU", "DA_TRA"):
            lost_query = lost_query.filter(LostItem.id == -1)

        lost_items = lost_query.order_by(LostItem.id.desc()).all()
    except Exception:
        lost_items = []

    try:
        found_query = FoundItem.query
        if keyword:
            found_query = found_query.filter(FoundItem.ten_do.ilike(f"%{keyword}%"))
        if category:
            found_query = found_query.filter(FoundItem.loai_do == category)
        if location:
            found_query = found_query.filter(FoundItem.noi_nhat.ilike(f"%{location}%"))
        if color:
            found_query = found_query.filter(FoundItem.mau_sac.ilike(f"%{color}%"))

        if status in ("DANG_LUU_GIU", "DA_TRA"):
            found_query = found_query.filter(FoundItem.trang_thai == status)
        elif status in ("DANG_TIM", "DA_TIM"):
            found_query = found_query.filter(FoundItem.id == -1)

        found_items = found_query.order_by(FoundItem.id.desc()).all()
    except Exception:
        found_items = []

    return render_template(
        "search.html",
        items=lost_items,
        lost_items=lost_items,
        found_items=found_items,
        keyword=keyword,
        category=category,
        location=location,
        color=color,
        status=status
    )


@app.route("/delete-lost/<int:item_id>")
@app.route("/delete/<int:item_id>")
def delete_item(item_id):
    try:
        item = LostItem.query.get_or_404(item_id)
        db.session.delete(item)
        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
    return redirect(url_for("list_items"))


@app.route("/edit-lost/<int:item_id>", methods=["GET", "POST"])
@app.route("/edit/<int:item_id>", methods=["GET", "POST"])
def edit_lost(item_id):
    try:
        item = LostItem.query.get_or_404(item_id)
    except Exception:
        return redirect(url_for("list_items"))

    if request.method == "POST":
        item.ten_do = request.form.get("ten_do")
        item.loai_do = request.form.get("loai_do")
        item.mau_sac = request.form.get("mau_sac")
        item.noi_mat = normalize_location_text(request.form.get("noi_mat", ""))
        item.ngay_mat = request.form.get("ngay_mat")

        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                item.hinh_anh = res.get("secure_url", "")
                if item.hinh_anh:
                    calculate_image_features(item.hinh_anh)
                    _persist_disk_cache()
            except Exception:
                pass

        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
        return redirect(url_for("list_items"))

    return render_template("edit_lost.html", item=item, is_lost=True)


@app.route("/edit-found/<int:item_id>", methods=["GET", "POST"])
def edit_found(item_id):
    try:
        item = FoundItem.query.get_or_404(item_id)
    except Exception:
        return redirect(url_for("found_list"))

    if request.method == "POST":
        item.ten_do = request.form.get("ten_do")
        item.loai_do = request.form.get("loai_do")
        item.mau_sac = request.form.get("mau_sac")
        item.noi_nhat = normalize_location_text(request.form.get("noi_nhat", ""))
        item.ngay_nhat = request.form.get("ngay_nhat")
        item.noi_luu_giu = normalize_location_text(request.form.get("noi_luu_giu", ""))

        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                item.hinh_anh = res.get("secure_url", "")
                if item.hinh_anh:
                    calculate_image_features(item.hinh_anh)
                    _persist_disk_cache()
            except Exception:
                pass

        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
        return redirect(url_for("found_list"))

    return render_template("edit_lost.html", item=item, is_lost=False)


@app.route("/dashboard")
def dashboard():
    try:
        tong_so = LostItem.query.count()
        all_items = LostItem.query.all()
    except Exception:
        tong_so = 0
        all_items = []

    CATEGORIES_MAP = {
        "Sách vở / Dụng cụ học tập": ["Sách vở / Dụng cụ học tập", "Sách vở", "Sách", "Vở", "Tập", "Bút", "Thước", "Bóp viết", "Sách Tin học", "Sách tin 11", "Dụng cụ học tập"],
        "Thiết bị điện tử (Laptop, Điện thoại...)": ["Thiết bị điện tử (Laptop, Điện thoại...)", "Thiết bị điện tử", "Laptop", "Điện thoại", "Máy tính Casio", "Casio", "mouse", "máy tính", "Chuột"],
        "Trang phục / Phụ kiện (Áo khoác, Ví...)": ["Trang phục / Phụ kiện (Áo khoác, Ví...)", "Trang phục / Phụ kiện", "Áo khoác", "Balo", "Mũ", "Ví", "Vali", "Cặp", "Nón"],
        "Giấy tờ / Thẻ học sinh": ["Giấy tờ / Thẻ học sinh", "Giấy tờ", "Thẻ học sinh", "Thẻ HS", "Căn cước", "Thẻ"],
        "Khác": ["Khác", "Bình nước", "Không xác định", "vase", "Khu vực khác"]
    }

    category_counts = {cat: 0 for cat in CATEGORIES_MAP.keys()}

    for item in all_items:
        raw_cat = str(item.loai_do).strip() if item.loai_do else "Khác"
        mapped = False
        for new_cat, old_list in CATEGORIES_MAP.items():
            if any(old.lower() in raw_cat.lower() or raw_cat.lower() in old.lower() for old in old_list):
                category_counts[new_cat] += 1
                mapped = True
                break
        if not mapped:
            category_counts["Khác"] += 1

    thong_ke = [(cat, count) for cat, count in category_counts.items()]
    return render_template("dashboard.html", tong_so=tong_so, thong_ke=thong_ke)


@app.route("/heatmap")
def heatmap():
    areas = [
        "Sân trường", "Nhà xe", "Căn tin", "Thư viện", "Sân bóng",
        "Phòng STEM", "Phòng Tin học 1", "Phòng Tin học 2",
        "Phòng Văn", "Phòng Sử - Địa", "Phòng Hội trường",
        "Phòng Đoàn - Đội", "Khu Văn phòng", "Phòng Tâm lý",
        "6A1", "6A2", "6A3", "7A1", "7A2", "7A3",
        "8A1", "8A2", "8A3", "9A1", "9A2", "9A3",
        "10A1", "10A2", "10A3", "11A1", "11A2", "12A1", "12A2"
    ]
    try:
        all_lost = LostItem.query.all()
        count_dict = {area: 0 for area in areas}
        unknown_count = 0
        for item in all_lost:
            loc = normalize_location_text(item.noi_mat)
            if loc in count_dict:
                count_dict[loc] += 1
            else:
                unknown_count += 1
    except Exception:
        count_dict = {area: 0 for area in areas}
        unknown_count = 0

    data = [(area, count_dict.get(area, 0)) for area in areas]
    if unknown_count > 0:
        data.append(("Chưa rõ khu vực", unknown_count))
    return render_template("heatmap.html", data=data)


@app.route("/found-return/<int:id>")
def found_return(id):
    try:
        item = FoundItem.query.get_or_404(id)
        item.trang_thai = "DA_TRA"
        now_vn = datetime.now(VN_TZ).replace(tzinfo=None)
        item.thoi_diem_tra = now_vn
        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
    return redirect(url_for("found_list"))


@app.route("/match")
@app.route("/match/<int:item_id>")
def match_items(item_id=None):
    """Đối sánh v39: Siêu nhẹ RAM (< 70MB) + Lưới chi tiết 8x8 + HOG + Sắc độ HSV."""
    page = request.args.get("page", 1, type=int)
    cat_filter = request.args.get("category", "").strip()
    per_page = 10

    try:
        lost_query = LostItem.query.filter(LostItem.trang_thai != "DA_TIM")
        found_query = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA")

        if item_id:
            lost_query = lost_query.filter(LostItem.id == item_id)
        if cat_filter:
            lost_query = lost_query.filter(LostItem.loai_do == cat_filter)
            found_query = found_query.filter(FoundItem.loai_do == cat_filter)

        lost_items = lost_query.order_by(LostItem.id.desc()).all()
        found_items = found_query.order_by(FoundItem.id.desc()).all()

        for obj in lost_items:
            obj.has_image = check_real_image(obj.hinh_anh)
            obj.noi_mat = normalize_location_text(obj.noi_mat)
        for obj in found_items:
            obj.has_image = check_real_image(obj.hinh_anh)
            obj.noi_nhat = normalize_location_text(obj.noi_nhat)
            obj.noi_luu_giu = normalize_location_text(obj.noi_luu_giu)

        candidates = []
        seen_signatures = set()
        sequence = 0

        for lost in lost_items:
            key1 = get_core_keyword(lost.ten_do)
            for found in found_items:
                try:
                    if (lost.loai_do and found.loai_do and
                            str(lost.loai_do).strip().lower() != str(found.loai_do).strip().lower()):
                        continue

                    sig = (
                        str(lost.ten_do or "").strip().lower(),
                        str(lost.noi_mat or "").strip().lower(),
                        str(lost.ngay_mat or "").strip(),
                        str(found.ten_do or "").strip().lower(),
                        str(found.noi_nhat or "").strip().lower(),
                        str(found.ngay_nhat or "").strip(),
                    )
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    key2 = get_core_keyword(found.ten_do)
                    ten_sim = similarity(lost.ten_do, found.ten_do)
                    is_same_key = (ten_sim >= 0.72) or (key1 == key2 and key1 != "")

                    if not is_same_key:
                        continue

                    ten_score = 15 if ten_sim >= 0.85 else int(ten_sim * 15)
                    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key)

                    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
                    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
                    base_score = ten_score + mau_score + vi_tri_score + ngay_score

                    has_both_images = 1 if (lost.has_image and found.has_image) else 0
                    non_conflict = 0 if color_conflict else 1

                    sequence += 1
                    candidates.append({
                        "lost": lost,
                        "found": found,
                        "base_score": base_score,
                        "has_both_images": has_both_images,
                        "non_conflict": non_conflict,
                        "is_same_key": is_same_key,
                        "color_conflict": color_conflict,
                        "score_name": ten_score,
                        "score_color": mau_score,
                        "score_location": vi_tri_score,
                        "score_date": ngay_score,
                        "seq": sequence
                    })
                except Exception:
                    continue

        candidates.sort(
            key=lambda x: (x["has_both_images"], x["non_conflict"], x["base_score"], x["seq"]),
            reverse=True
        )
        total_matches = len(candidates)
        total_pages = max(1, (total_matches + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        page_candidates = candidates[start_idx:start_idx + per_page]

        urls_to_fetch = set()
        for item in page_candidates:
            if item["lost"].has_image and item["found"].has_image:
                if get_feature_cache(item["lost"].hinh_anh) is None:
                    urls_to_fetch.add(item["lost"].hinh_anh)
                if get_feature_cache(item["found"].hinh_anh) is None:
                    urls_to_fetch.add(item["found"].hinh_anh)

        if urls_to_fetch:
            with ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(calculate_image_features, urls_to_fetch))
            _persist_disk_cache()

        page_matches = []
        for item in page_candidates:
            lost, found = item["lost"], item["found"]
            v_cols1, v_cols2 = None, None
            if lost.has_image and found.has_image:
                f1 = get_feature_cache(lost.hinh_anh)
                f2 = get_feature_cache(found.hinh_anh)
                v_cols1 = f1.get("visual_colors") if f1 else None
                v_cols2 = f2.get("visual_colors") if f2 else None

            mau_score, color_conflict = evaluate_color_match(
                lost, found, item["is_same_key"], v_cols1, v_cols2
            )

            if lost.has_image and found.has_image:
                anh_score, _ = image_score(
                    lost.hinh_anh, found.hinh_anh,
                    is_same_keyword=item["is_same_key"],
                    color_conflict=color_conflict
                )
            else:
                anh_score = 0

            total_score = item["score_name"] + mau_score + item["score_location"] + item["score_date"] + anh_score
            if total_score >= 75:
                ket_luan = "🟢 Khả năng trùng khớp cao"
            elif total_score >= 55:
                ket_luan = "🟡 Có khả năng trùng khớp"
            elif total_score >= 38:
                ket_luan = "🟠 Có một số điểm tương đồng"
            else:
                ket_luan = "🔴 Khả năng trùng khớp thấp"

            page_matches.append({
                "lost": lost,
                "found": found,
                "score": total_score,
                "ket_luan": ket_luan,
                "score_image": anh_score,
                "score_name": item["score_name"],
                "score_color": mau_score,
                "score_location": item["score_location"],
                "score_date": item["score_date"],
            })

        page_matches.sort(key=lambda x: x["score"], reverse=True)
        gc.collect()

        return render_template(
            "match.html",
            matches=page_matches,
            page=page,
            total_pages=total_pages,
            total_matches=total_matches,
            selected_category=cat_filter
        )
    except Exception as route_err:
        print(f"Lỗi route /match: {route_err}")
        gc.collect()
        return render_template(
            "match.html",
            matches=[],
            page=1,
            total_pages=1,
            total_matches=0,
            selected_category=cat_filter
        )


@app.route("/teacher-check/<int:lost_id>/<int:found_id>")
def teacher_check(lost_id, found_id):
    try:
        lost = LostItem.query.get_or_404(lost_id)
        found = FoundItem.query.get_or_404(found_id)
        lost.noi_mat = normalize_location_text(lost.noi_mat)
        found.noi_nhat = normalize_location_text(found.noi_nhat)
        found.noi_luu_giu = normalize_location_text(found.noi_luu_giu)
    except Exception:
        return redirect(url_for("match_items"))

    has1 = check_real_image(lost.hinh_anh)
    has2 = check_real_image(found.hinh_anh)

    key1 = get_core_keyword(lost.ten_do)
    key2 = get_core_keyword(found.ten_do)
    ten_sim = similarity(lost.ten_do, found.ten_do)
    is_same_key = (ten_sim >= 0.72) or (key1 == key2 and key1 != "")

    ten_score = 15 if ten_sim >= 0.85 else int(ten_sim * 15)

    v_cols1, v_cols2 = None, None
    if has1 and has2:
        f1 = calculate_image_features(lost.hinh_anh)
        f2 = calculate_image_features(found.hinh_anh)
        v_cols1 = f1.get("visual_colors") if f1 else None
        v_cols2 = f2.get("visual_colors") if f2 else None

    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key, v_cols1, v_cols2)
    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

    if has1 and has2:
        anh_score, image_ratio = image_score(
            lost.hinh_anh, found.hinh_anh,
            is_same_keyword=is_same_key,
            color_conflict=color_conflict
        )
        _persist_disk_cache()
    else:
        anh_score, image_ratio = 0, 0.0

    score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score

    if score >= 75:
        ket_luan = "🟢 Khả năng trùng khớp cao"
    elif score >= 55:
        ket_luan = "🟡 Có khả năng trùng khớp"
    elif score >= 38:
        ket_luan = "🟠 Có một số điểm tương đồng"
    else:
        ket_luan = "🔴 Khả năng trùng khớp thấp"

    image_warning = None
    if color_conflict:
        image_warning = "⚠️ Hai món đồ có màu sắc trái ngược nhau, giáo viên cần đối chiếu kỹ."
    elif has1 and has2 and image_ratio < 0.25:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."
    elif not (has1 and has2):
        image_warning = "⚠️ Bài đăng này thiếu hình ảnh minh họa, vui lòng xác minh bằng thông tin mô tả."

    gc.collect()
    return render_template(
        "teacher_check.html",
        lost=lost,
        found=found,
        score=score,
        ket_luan=ket_luan,
        image_warning=image_warning
    )


@app.route("/confirm-return/<int:lost_id>/<int:found_id>", methods=["GET", "POST"])
def confirm_return(lost_id, found_id):
    pin_code = request.form.get("pin_code", "").strip() if request.method == "POST" else request.args.get("pin_code", "").strip()
    current_pin = get_current_pin()

    if pin_code != current_pin:
        return """
        <script>
            alert("Mã PIN Giáo viên không chính xác! Vui lòng thử lại.");
            window.history.back();
        </script>
        """

    try:
        lost = LostItem.query.get_or_404(lost_id)
        found = FoundItem.query.get_or_404(found_id)

        found.trang_thai = "DA_TRA"
        lost.trang_thai = "DA_TIM"

        thoi_diem = datetime.now(VN_TZ).replace(tzinfo=None)
        found.thoi_diem_tra = thoi_diem
        lost.thoi_diem_tra = thoi_diem

        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass

    return """
    <script>
        alert("Xác nhận đã trả thành công! Bài báo mất đã chuyển sang trạng thái 'Đã tìm thấy'.");
        window.location.href = "/lost-list";
    </script>
    """


@app.route("/export-excel", methods=["GET"])
def export_excel():
    try:
        status = request.args.get("status", "all")
        month = request.args.get("month", "")
        year = request.args.get("year", "")

        lost_query = LostItem.query
        found_query = FoundItem.query

        status_label = "Tat_Ca"
        if status == "completed":
            lost_query = lost_query.filter(LostItem.trang_thai.in_(["DA_TIM", "Đã tìm thấy"]))
            found_query = found_query.filter(FoundItem.trang_thai.in_(["DA_TRA", "Đã hoàn trả"]))
            status_label = "Da_Hoan_Tra"
        elif status == "pending":
            lost_query = lost_query.filter(LostItem.trang_thai.in_(["DANG_TIM", "Đang tìm"]))
            found_query = found_query.filter(FoundItem.trang_thai.in_(["DANG_LUU_GIU", "Đang lưu giữ"]))
            status_label = "Dang_Luu_Giu"

        time_label = "Toan_Thoi_Gian"
        if year:
            lost_query = lost_query.filter(LostItem.ngay_mat.like(f"{year}-%"))
            found_query = found_query.filter(FoundItem.ngay_nhat.like(f"{year}-%"))
            time_label = f"Nam_{year}"

        if month:
            m_str = f"{int(month):02d}"
            lost_query = lost_query.filter(LostItem.ngay_mat.like(f"%-{m_str}-%"))
            found_query = found_query.filter(FoundItem.ngay_nhat.like(f"%-{m_str}-%"))
            time_label = f"Thang_{m_str}_{year if year else ''}".strip("_")

        lost_items = lost_query.all()
        found_items = found_query.all()

        lost_data = [{
            "STT": i + 1,
            "Tên đồ vật": item.ten_do,
            "Loại đồ": item.loai_do,
            "Màu sắc": item.mau_sac,
            "Nơi mất": normalize_location_text(item.noi_mat),
            "Ngày mất": format_vn_date(item.ngay_mat),
            "Trạng thái": "Đã tìm thấy" if item.trang_thai in ["DA_TIM", "Đã tìm thấy"] else "Đang tìm"
        } for i, item in enumerate(lost_items)]

        found_data = [{
            "STT": i + 1,
            "Tên đồ vật": item.ten_do,
            "Loại đồ": item.loai_do,
            "Màu sắc": item.mau_sac,
            "Nơi nhặt": normalize_location_text(item.noi_nhat),
            "Ngày nhặt": format_vn_date(item.ngay_nhat),
            "Nơi lưu giữ / Liên hệ": normalize_location_text(item.noi_luu_giu),
            "Trạng thái": "Đã hoàn trả" if item.trang_thai in ["DA_TRA", "Đã hoàn trả"] else "Đang lưu giữ"
        } for i, item in enumerate(found_items)]

        df_lost = pd.DataFrame(lost_data)
        df_found = pd.DataFrame(found_data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_lost.to_excel(writer, sheet_name="Đồ báo mất", index=False)
            df_found.to_excel(writer, sheet_name="Đồ báo nhặt", index=False)

        output.seek(0)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BaoCao_DoThatLac_{status_label}_{time_label}_{now_str}.xlsx"

        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename
        )
    except Exception:
        return "Lỗi xuất Excel", 500


@app.route("/get-pin")
def get_pin():
    return jsonify({"pin": get_current_pin()})


@app.route("/change-pin", methods=["GET", "POST"])
def change_pin():
    if request.method == "POST":
        old_pin = request.form.get("old_pin", "").strip()
        new_pin = request.form.get("new_pin", "").strip()
        confirm_pin = request.form.get("confirm_pin", "").strip()

        current_pin = get_current_pin()

        if old_pin != current_pin:
            return render_template("change_pin.html", error="Mã PIN cũ không chính xác!")

        if new_pin != confirm_pin:
            return render_template("change_pin.html", error="Mã PIN mới và Xác nhận PIN không trùng khớp!")

        if len(new_pin) < 4:
            return render_template("change_pin.html", error="Mã PIN mới phải có ít nhất 4 ký tự!")

        if save_new_pin(new_pin):
            return render_template("change_pin.html", success=f"Đổi Mã PIN thành công! Mã PIN mới là: {new_pin}")
        else:
            return render_template("change_pin.html", error="Lỗi không thể lưu mã PIN mới. Vui lòng thử lại!")

    return render_template("change_pin.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
