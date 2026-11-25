import os
import io
import uuid
import json
import re
import gc
import math
import colorsys
import threading
import unicodedata
import urllib3
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory, session, flash
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

HAS_ONNX = False
try:
    import onnxruntime as ort
    HAS_ONNX = True
except Exception as e:
    print(f"Thông báo ONNX Runtime chưa cài đặt (dùng chế độ thị giác nhẹ): {e}")

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


def strip_accents(text_val):
    """Chuyển tiếng Việt có dấu sang không dấu viết thường (đ -> d, ă -> a...)."""
    if not text_val:
        return ""
    s = str(text_val).strip().lower()
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


@app.template_filter("vn_date")
def format_vn_date(date_val):
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


@app.template_filter("mask_hint")
def mask_distinctive_hint(note_val):
    """
    Hiển thị gợi ý dấu hiệu riêng cho học sinh dễ nhận ra đồ,
    đồng thời che bớt mã lớp và tên riêng (kể cả khi viết thường sau từ 'tên', 'chữ', 'ký', 'ghi'...).
    """
    if not note_val:
        return ""
    s = str(note_val).strip()
    if s.lower() in {"none", "null", ""}:
        return ""

    # 1. Che mã lớp (ví dụ: 7A3 -> 7**, 11A1 -> 11**)
    s = re.sub(r'\b(\d{1,2})[a-zA-Z]\d{1,2}\b', r'\1**', s)

    # 2. Che tên riêng viết hoa hoặc từ đứng ngay sau từ khóa chỉ tên/ký hiệu
    trigger_words = {"ten", "chu", "ky", "ki", "ghi", "de"}
    ignore_after_trigger = {"la", "o", "bang", "muc", "but", "nho", "to", "mo", "ro", "lop", "rieng"}

    tokens = s.split(" ")
    masked_tokens = []
    prev_is_trigger = False

    for idx, tok in enumerate(tokens):
        clean_alpha = re.sub(r'[^A-Za-zÀ-ỹ]', '', tok)
        punct_suffix = tok[len(clean_alpha):] if clean_alpha and tok.startswith(clean_alpha) else ""
        low_alpha = strip_accents(clean_alpha)

        should_mask = False
        if len(clean_alpha) >= 2:
            if prev_is_trigger and low_alpha not in ignore_after_trigger:
                should_mask = True
            elif idx > 0 and clean_alpha[0].isupper():
                should_mask = True

        if should_mask:
            masked_tokens.append(clean_alpha[0] + "***" + punct_suffix)
        else:
            masked_tokens.append(tok)

        prev_is_trigger = (low_alpha in trigger_words)

    return " ".join(masked_tokens)


UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
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


class SystemConfig(db.Model):
    __tablename__ = "system_config"
    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.String(255), nullable=False)


with app.app_context():
    try:
        db.create_all()
        for sql_cmd in [
            "ALTER TABLE found_item ADD COLUMN noi_luu_giu VARCHAR(200);",
            "ALTER TABLE found_item ADD COLUMN thoi_diem_tra TIMESTAMP;",
            "ALTER TABLE lost_item ADD COLUMN thoi_diem_tra TIMESTAMP;",
            "ALTER TABLE found_item ADD COLUMN mo_ta TEXT;",
            "ALTER TABLE lost_item ADD COLUMN mo_ta TEXT;",
            "ALTER TABLE found_item ALTER COLUMN hinh_anh TYPE TEXT;",
            "ALTER TABLE lost_item ALTER COLUMN hinh_anh TYPE TEXT;"
        ]:
            try:
                db.session.execute(text(sql_cmd))
                db.session.commit()
            except Exception:
                db.session.rollback()
    except Exception as e:
        db.session.rollback()
        print(f"Thông báo DB migration: {e}")

CACHE_VERSION = 58
CACHE_FILE = f"feature_cache_mobilenet_v{CACHE_VERSION}.json"
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
                if isinstance(v, dict) and v.get("version") == CACHE_VERSION
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

# ==============================================================================
# MOBILENETV2 ONNX ENGINE
# ==============================================================================
MOBILENET_MODEL_PATH = os.path.join("ai", "mobilenetv2-12.onnx")
MOBILENET_URLS = [
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx",
    "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx"
]
_ORT_SESSION = None
_ORT_LOCK = threading.Lock()


def _get_mobilenet_session():
    global _ORT_SESSION
    if not (HAS_ONNX and HAS_NUMPY):
        return None
    if _ORT_SESSION is not None:
        return _ORT_SESSION

    with _ORT_LOCK:
        if _ORT_SESSION is not None:
            return _ORT_SESSION
        try:
            os.makedirs("ai", exist_ok=True)
            if not os.path.isfile(MOBILENET_MODEL_PATH) or os.path.getsize(MOBILENET_MODEL_PATH) < 1_000_000:
                for url in MOBILENET_URLS:
                    try:
                        resp = HTTP_SESSION.get(url, timeout=(5.0, 25.0), verify=False)
                        if resp.status_code == 200 and len(resp.content) > 1_000_000:
                            with open(MOBILENET_MODEL_PATH, "wb") as f:
                                f.write(resp.content)
                            break
                    except Exception:
                        continue

            if os.path.isfile(MOBILENET_MODEL_PATH) and os.path.getsize(MOBILENET_MODEL_PATH) > 1_000_000:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.enable_mem_pattern = False
                _ORT_SESSION = ort.InferenceSession(
                    MOBILENET_MODEL_PATH,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                print("✅ Đã khởi động thành công MobileNetV2 ONNX!")
        except Exception as e:
            print(f"Không thể nạp MobileNetV2 ONNX: {e}")
            _ORT_SESSION = None
    return _ORT_SESSION


def _extract_mobilenet_embedding(pil_img):
    sess = _get_mobilenet_session()
    if sess is None:
        return None, []
    try:
        w, h = pil_img.size
        mx, my = int(w * 0.10), int(h * 0.10)
        if (w - 2 * mx) >= 32 and (h - 2 * my) >= 32:
            crop = pil_img.crop((mx, my, w - mx, h - my))
        else:
            crop = pil_img

        img_224 = crop.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(img_224, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]

        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: arr})[0].flatten().astype(np.float32)

        top_ids = [int(i) for i in np.argsort(logits)[-8:][::-1]]

        shifted = logits - float(np.max(logits))
        probs = np.exp(shifted * 0.5)
        norm = float(np.linalg.norm(probs))
        if norm > 1e-6:
            probs = probs / norm
        return [round(float(x), 5) for x in probs], top_ids
    except Exception as e:
        print(f"Lỗi suy luận MobileNetV2: {e}")
        return None, []


STOP_WORDS_NO_ACCENT = {
    "con", "cai", "chiec", "hop", "bo", "cap", "doi", "qua", "trai",
    "tam", "buc", "to", "cuc", "quyen", "cuon", "cay", "ve", "mau",
    "loai", "de", "ban", "hoc", "do", "dung", "cu", "thiet", "bi", "tap", "vat"
}
COLOR_WORDS_MAP = {
    "den": "đen", "do": "đỏ", "xanh": "xanh", "vang": "vàng",
    "trang": "trắng", "hong": "hồng", "tim": "tím", "nau": "nâu",
    "xam": "xám", "cam": "cam", "bac": "bạc"
}

MARK_SYNONYM_GROUPS = {
    "hu_khoa": ["hong khoa", "hu khoa", "ket khoa", "gay khoa", "mat dau khoa", "tuot khoa", "day keo hong", "day keo ket"],
    "rach_son": ["rach", "thung", "son", "sut chi", "bung chi", "tray", "xuoc", "me", "nut", "bi vo", "vo nat"],
    "vet_ban": ["dinh muc", "vet muc", "ban", "o", "dinh son", "but xoa", "vet ban", "lem muc"],
    "mat_cuc": ["mat cuc", "rung cuc", "mat nut", "rung nut", "dut cuc", "thieu cuc", "thieu nut"],
    "tem_mac": ["mac", "tem", "nhan", "co ao", "vat ao", "nap may", "mat sau", "ben trong"],
    "moc_khoa": ["moc khoa", "gau bong", "day deo", "sticker", "hinh dan", "nhan dan", "bang ten"],
    "vi_tri_chi_tiet": ["tay trai", "tay phai", "tui ao", "tui trai", "tui phai", "lung ao", "goc duoi", "goc tren", "ngan ngoai", "ngan trong"],

CATEGORIES_MAP = {
    "Sách vở / Dụng cụ học tập": [
        "Sách vở / Dụng cụ học tập", "Sách vở", "Sách", "Vở", "Tập", "Bút",
        "Thước", "Compa", "Bóp viết", "Hộp bút", "Sách Tin học", "Sách tin 11",
        "Dụng cụ học tập", "Đồ dùng học tập", "Dụng cụ học toán", "Đồ học toán"
    ],
    "Thiết bị điện tử (Laptop, Điện thoại...)": [
        "Thiết bị điện tử (Laptop, Điện thoại...)", "Thiết bị điện tử", "Laptop",
        "Điện thoại", "Máy tính Casio", "Casio", "mouse", "máy tính", "Chuột",
        "Tai nghe", "Sạc dự phòng", "Đồng hồ thông minh"
    ],
    "Trang phục / Phụ kiện (Áo khoác, Ví...)": [
        "Trang phục / Phụ kiện (Áo khoác, Ví...)", "Trang phục / Phụ kiện",
        "Áo khoác", "Balo", "Ba lô", "Mũ", "Ví", "Bóp", "Vali", "Cặp", "Nón", "Kính"
    ],
    "Giấy tờ / Thẻ học sinh": [
        "Giấy tờ / Thẻ học sinh", "Giấy tờ", "Thẻ học sinh", "Thẻ HS", "Căn cước", "Thẻ", "CCCD"
    ],
    "Khác": [
        "Khác", "Bình nước", "Chai nước", "Ly", "Không xác định", "vase", "Khu vực khác"
    ]
}

IMAGENET_TO_SCHOOL_CAT = [
    ("Thiết bị điện tử (Laptop, Điện thoại...)", {
        508, 527, 620, 664, 681, 782, 851,
        487, 605, 761, 892, 531, 696,
    }),
    ("Trang phục / Phụ kiện (Áo khoác, Ví...)", {
        414, 452, 474, 501, 515, 568, 601, 610, 617, 655, 678, 735, 824, 834, 841, 869,
        636, 748, 893,
    }),
    ("Sách vở / Dụng cụ học tập", {
        418, 449, 454, 496, 549, 557, 567, 710, 767, 796,
    }),
    ("Giấy tờ / Thẻ học sinh", {
        916, 917, 921,
    }),
]


def _predict_category_from_ids(top_ids):
    if not top_ids:
        return ""
    id_set = set(top_ids)
    for cat_name, target_ids in IMAGENET_TO_SCHOOL_CAT:
        if id_set.intersection(target_ids):
            return cat_name
    return ""


def auto_normalize_category(ten_do, selected_loai_do):
    raw_name = strip_accents(ten_do)
    raw_cat = str(selected_loai_do or "").strip()

    explicit_rules = [
        ("Trang phục / Phụ kiện (Áo khoác, Ví...)", ["balo", "ba lo", "ao khoac", "ao am", "vi ", "bop ", "mu ", "non ", "vali", "cap sach"]),
        ("Thiết bị điện tử (Laptop, Điện thoại...)", ["casio", "fx580", "fx 580", "880btg", "laptop", "dien thoai", "iphone", "samsung", "chuot may", "tai nghe", "dong ho thong minh"]),
        ("Giấy tờ / Thẻ học sinh", ["the hoc sinh", "the hs", "can cuoc", "cccd", "giay to", "chung minh"]),
        ("Sách vở / Dụng cụ học tập", ["sach ", "vo ", "tap ", "but ", "thuoc", "compa", "hop but", "bop viet", "dung cu", "do dung hoc", "do hoc", "xac suat", "thiet bi"]),
        ("Khác", ["binh nuoc", "chai nuoc", "binh giu nhiet", "chia khoa", "o ", "du "]),
    ]
    for target_cat, keywords in explicit_rules:
        if any(kw in f"{raw_name} " for kw in keywords):
            return target_cat
    return raw_cat


def normalize_location_text(loc):
    if not loc:
        return ""
    s = str(loc).strip()
    if s.lower() in {"none", "null", ""}:
        return ""
    return re.sub(r'căng\s*tin', 'Căn tin', s, flags=re.IGNORECASE)


def clean_note_text(note):
    if not note:
        return ""
    s = str(note).strip()
    return "" if s.lower() in {"none", "null"} else s


def extract_colors(text):
    if not text:
        return set()
    t = strip_accents(text)
    found = set()
    for no_acc, vn_col in COLOR_WORDS_MAP.items():
        if re.search(rf'\b{no_acc}\b', t):
            found.add(vn_col)
    if "bạc" in found or "trắng" in found:
        found.update({"trắng", "bạc"})
    return found


def _extract_mark_entities(raw_text):
    text = clean_note_text(raw_text)
    if not text:
        return {"names": set(), "codes": set(), "concepts": set(), "words": set()}

    low_no_acc = strip_accents(text)
    names = set()

    for m in re.finditer(r'(?:ten|chu|ky|ki|ghi|de)\s+([a-z]{2,15})', low_no_acc):
        cand = m.group(1).strip()
        if cand not in {"la", "o", "bang", "muc", "but", "nho", "to", "mo", "ro", "lop"}:
            names.add(cand)

    tokens_raw = re.findall(r'[A-Za-zÀ-ỹ]+', text)
    for idx, tok in enumerate(tokens_raw):
        if idx > 0 and tok[0].isupper() and len(tok) >= 2:
            low_tok = strip_accents(tok)
            if low_tok not in STOP_WORDS_NO_ACCENT and low_tok not in COLOR_WORDS_MAP:
                names.add(low_tok)

    codes = {c.lower() for c in re.findall(r'\b(?:\d{1,2}[a-zA-Z]\d{1,2}|\d{2,6})\b', low_no_acc)}

    concepts = set()
    for group_key, phrases in MARK_SYNONYM_GROUPS.items():
        if any(p in low_no_acc for p in phrases):
            concepts.add(group_key)

    ignore_words = STOP_WORDS_NO_ACCENT | {
        "co", "bi", "o", "trong", "ngoai", "tren", "duoi", "mot", "hai", "rat",
        "hoi", "nho", "lon", "phia", "ben", "ghi", "viet", "ten", "chu", "lop", "ao"
    }
    clean_words = {
        w for w in re.sub(r'[^\w\s]', ' ', low_no_acc).split()
        if len(w) >= 2 and w not in ignore_words
    }

    return {
        "names": names,
        "codes": codes,
        "concepts": concepts,
        "words": clean_words
    }


def analyze_distinctive_marks(lost_item, found_item):
    """
    Phân tích và trích xuất các chi tiết đối chứng từ Dấu hiệu nhận dạng riêng
    để hiển thị cho Giáo viên xác minh (không cộng/trừ vào thang điểm 100).
    """
    m1 = clean_note_text(getattr(lost_item, "mo_ta", ""))
    m2 = clean_note_text(getattr(found_item, "mo_ta", ""))
    if not m1 or not m2:
        return []

    e1 = _extract_mark_entities(m1)
    e2 = _extract_mark_entities(m2)
    hints = []

    common_names = e1["names"].intersection(e2["names"])
    if common_names:
        hints.append(f"Trùng tên riêng: {', '.join(sorted(common_names)).title()}")

    common_codes = e1["codes"].intersection(e2["codes"])
    if common_codes:
        hints.append(f"Trùng mã/lớp: {', '.join(sorted(common_codes)).upper()}")

    common_concepts = e1["concepts"].intersection(e2["concepts"])
    if common_concepts:
        hints.append("Trùng đặc điểm chi tiết")

    return hints


def split_image_list(img_field):
    if not img_field:
        return []
    parts = [p.strip() for p in str(img_field).split(",") if p.strip()]
    valid = []
    invalid = ["chua co anh", "chưa có ảnh", "no_image", "placeholder", "120?text="]
    for p in parts:
        low = p.lower()
        if low in {"none", "null", "undefined"} or any(tok in low for tok in invalid):
            continue
        valid.append(p)
    return valid


def cleanup_local_images(img_field):
    for img in split_image_list(img_field):
        if not img.lower().startswith(("http://", "https://")):
            local_path = img if os.path.isabs(img) else os.path.join(
                app.config.get("UPLOAD_FOLDER", "static/uploads"), img
            )
            try:
                if os.path.isfile(local_path):
                    os.remove(local_path)
            except Exception as e:
                print(f"Không thể xóa file ảnh cục bộ {local_path}: {e}")


def upload_multiple_images(file_list, max_images=3):
    saved_urls = []
    valid_files = [f for f in file_list if f and f.filename != ""][:max_images]
    for file in valid_files:
        url = ""
        try:
            res = cloudinary.uploader.upload(file)
            url = res.get("secure_url", "")
        except Exception as cloud_err:
            print("Chuyển sang lưu ảnh cục bộ do:", cloud_err)
            file.seek(0)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            url = f"{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], url))
        if url:
            saved_urls.append(url)
            try:
                calculate_image_features(url)
            except Exception as feat_err:
                print("Bỏ qua tính feature server:", feat_err)
    if saved_urls:
        try:
            _persist_disk_cache()
        except Exception:
            pass
    return ",".join(saved_urls)


def check_real_image(img_path):
    imgs = split_image_list(img_path)
    if not imgs:
        return False
    for value in imgs:
        low = value.lower()
        if low.startswith(("http://", "https://")):
            return True
        path = value if os.path.isabs(value) else os.path.join(
            app.config.get("UPLOAD_FOLDER", "static/uploads"), value
        )
        if os.path.isfile(path):
            return True
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
    value = str(img_path).strip()
    if "res.cloudinary.com/" in value and "/image/upload/" in value and "/w_" not in value:
        value = value.replace("/image/upload/", "/image/upload/w_256,c_limit,q_auto:low,f_jpg/", 1)
    if value.startswith(("http://", "https://")):
        response = HTTP_SESSION.get(value, timeout=(2.5, 5.0), verify=False)
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
    w0, h0 = pil_img.size
    mx, my = int(w0 * 0.12), int(h0 * 0.12)
    center_img = pil_img.crop((mx, my, w0 - mx, h0 - my)) if (w0 - 2 * mx >= 16 and h0 - 2 * my >= 16) else pil_img

    size = 48
    small = center_img.resize((size, size), Image.Resampling.BILINEAR)
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
    bg_palette = border_samples[::3]

    mask = [False] * (size * size)
    min_x, min_y, max_x, max_y = size, size, 0, 0
    fg_count = 0

    for y in range(1, size - 1):
        row_off = y * size
        for x in range(1, size - 1):
            idx = row_off + x
            r, g, b = px[idx]
            _, sat, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            min_bg_dist = 999999
            for br, bg_c, bb in bg_palette:
                dist_sq = (r - br) ** 2 + (g - bg_c) ** 2 + (b - bb) ** 2
                if dist_sq < min_bg_dist:
                    min_bg_dist = dist_sq
                    if min_bg_dist < 450:
                        break
            if min_bg_dist > 700 or (abs(sat - bg_sat) > 0.12 and min_bg_dist > 280):
                mask[idx] = True
                fg_count += 1
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)

    if fg_count < 16 or max_x <= min_x + 2 or max_y <= min_y + 2:
        full_mask = [(4 <= (i % size) < 44 and 4 <= (i // size) < 44) for i in range(size * size)]
        return center_img, small, full_mask

    cw, ch = center_img.size
    x0 = max(0, int(((min_x - 2) / size) * cw))
    y0 = max(0, int(((min_y - 2) / size) * ch))
    x1 = min(cw, int(((max_x + 2) / size) * cw))
    y1 = min(ch, int(((max_y + 2) / size) * ch))
    return center_img.crop((x0, y0, max(x0 + 4, x1), max(y0 + 4, y1))), small, mask


def _detect_visual_colors_masked(small_48, fg_mask):
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
        detected = {c for c, cnt in counts.items() if (cnt / total) >= 0.15}
        if "trắng" in detected or "xám" in detected:
            detected.add("bạc")
        return detected
    except Exception:
        return set()


def _compute_masked_color_hist(small_48, fg_mask):
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
        weight = 1.35 if s >= 0.30 else 0.85
        if s < 0.16:
            bins[min(3, int(v * 4.0))] += weight
        else:
            h_pos = (h * 12.0) % 12.0
            h0 = int(h_pos)
            frac = h_pos - h0
            bins[4 + h0] += weight * (1.0 - frac * 0.45)
            bins[4 + ((h0 + 1) % 12)] += weight * (frac * 0.45)
        rgb_sum = float(rv + gv + bv)
        if rgb_sum > 30.0:
            r_bin = min(3, int((rv / rgb_sum) * 4.0))
            g_bin = min(2, int((gv / rgb_sum) * 3.0))
            bins[16 + r_bin * 3 + g_bin] += 0.8 * weight
        count += weight

    if count <= 0.0:
        return [1.0 / 28.0] * 28
    inv = 1.0 / count
    return [x * inv for x in bins]


def calculate_image_features(img_path):
    if not check_real_image(img_path):
        return None

    cached = get_feature_cache(img_path)
    if cached is not None and cached.get("version") == CACHE_VERSION:
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

            emb_orig, top_ids_orig = _extract_mobilenet_embedding(image)
            emb_flip, top_ids_flip = (
                _extract_mobilenet_embedding(ImageOps.mirror(image))
                if emb_orig is not None else (None, [])
            )
            combined_top_ids = sorted(set(top_ids_orig + top_ids_flip))

            cw, ch = fg_crop.size
            aspect_ratio = min(cw, ch) / max(1.0, max(cw, ch))

            features = {
                "version": CACHE_VERSION,
                "mobilenet_emb": emb_orig,
                "mobilenet_flip": emb_flip,
                "top_class_ids": combined_top_ids,
                "fg_color_hist": fg_color_hist,
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


def _single_pair_image_score(u1, u2, is_same_keyword=False, is_same_cat=False, color_conflict=False):
    """Chấm điểm hình ảnh trên thang chuẩn 0–50 điểm (Màu vật thể 0–18đ + MobileNetV2 0–32đ)."""
    if u1 == u2:
        return 50, 1.0, True

    f1 = calculate_image_features(u1)
    f2 = calculate_image_features(u2)
    if f1 is None or f2 is None:
        return 0, 0.0, False

    h1 = f1.get("fg_color_hist", [])
    h2 = f2.get("fg_color_hist", [])
    color_inter = sum(min(a, b) for a, b in zip(h1, h2)) if (h1 and h2) else 0.0
    color_cos = _cosine_sim(h1, h2) if (h1 and h2) else 0.0
    color_blend = 0.50 * color_inter + 0.50 * color_cos
    color_pts = max(2, min(18, int(round(((color_blend - 0.15) / 0.45) * 18))))

    e1 = f1.get("mobilenet_emb")
    e2 = f2.get("mobilenet_emb")
    e2_flip = f2.get("mobilenet_flip")
    ids1 = set(f1.get("top_class_ids", []))
    ids2 = set(f2.get("top_class_ids", []))
    shared_visual_classes = len(ids1.intersection(ids2))
    visually_same_object = shared_visual_classes >= 2

    if e1 and e2:
        sim_orig = _cosine_sim(e1, e2)
        sim_flip = _cosine_sim(e1, e2_flip) if e2_flip else sim_orig
        deep_sim = max(sim_orig, sim_flip)
        deep_pts = max(6, min(32, int(round(((deep_sim - 0.25) / 0.55) * 32))))
        if shared_visual_classes >= 3:
            deep_pts = min(32, deep_pts + 4)
    else:
        ar1 = float(f1.get("aspect_ratio", 0.7))
        ar2 = float(f2.get("aspect_ratio", 0.7))
        ar_sim = 1.0 - min(1.0, abs(ar1 - ar2))
        deep_pts = int(round(10 + ar_sim * 16))

    total = color_pts + deep_pts

    if color_conflict and total < 44:
        total = int(round(total * 0.80))

    total = max(8, min(49, total))

    if not is_same_keyword and not visually_same_object:
        cap = 26 if is_same_cat else 18
        if total > cap:
            total = cap

    return int(total), float(total / 50.0), visually_same_object


def image_score(
    img1_filename,
    img2_filename,
    is_same_keyword=False,
    is_same_cat=False,
    color_conflict=False
):
    try:
        list1 = split_image_list(img1_filename)
        list2 = split_image_list(img2_filename)
        if not list1 or not list2:
            return 0, 0.0, 0, 0, False

        best_score = 0
        best_ratio = 0.0
        best_i1 = 0
        best_i2 = 0
        best_vis_same = False
        for i1, u1 in enumerate(list1):
            for i2, u2 in enumerate(list2):
                sc, rt, vis_same = _single_pair_image_score(
                    u1, u2,
                    is_same_keyword=is_same_keyword,
                    is_same_cat=is_same_cat,
                    color_conflict=color_conflict
                )
                if sc > best_score:
                    best_score = sc
                    best_ratio = rt
                    best_i1 = i1
                    best_i2 = i2
                    best_vis_same = vis_same
        return best_score, best_ratio, best_i1, best_i2, best_vis_same
    except Exception as exc:
        print(f"Lỗi tính điểm hình ảnh: {exc}")
        return 0, 0.0, 0, 0, False


def _tokenize_meaningful_words(text_name):
    if not text_name:
        return set()
    t = re.sub(r'[^\w\s]', ' ', strip_accents(text_name))
    words = [
        w for w in t.split()
        if len(w) >= 2 and w not in STOP_WORDS_NO_ACCENT and w not in COLOR_WORDS_MAP
    ]
    return set(words)


def get_core_keyword(text_name):
    if not text_name:
        return ""
    t = strip_accents(text_name)

    if "chuot" in t or "mouse" in t:
        return "chuot"

    tokens = sorted(_tokenize_meaningful_words(t))
    if not tokens:
        return t
    nums = re.findall(r'\d+', t)
    base = tokens[0]
    return f"{base}_{'_'.join(nums)}" if nums else base


def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = strip_accents(text1), strip_accents(text2)

    if t1 == t2:
        return 1.0

    nums1 = re.findall(r'\d+', t1)
    nums2 = re.findall(r'\d+', t2)
    if nums1 and nums2 and nums1 != nums2:
        return 0.0

    if ("chuot" in t1) != ("chuot" in t2):
        return 0.0

    k1 = get_core_keyword(t1)
    k2 = get_core_keyword(t2)
    if k1 and k2 and k1 == k2:
        return 0.95

    w1 = _tokenize_meaningful_words(t1)
    w2 = _tokenize_meaningful_words(t2)
    common = w1.intersection(w2)
    if common:
        non_generic = {w for w in common if w not in {"sach", "vo", "may"}}
        if non_generic:
            overlap_ratio = len(non_generic) / max(1, min(len(w1), len(w2)))
            return min(0.95, 0.78 + 0.17 * overlap_ratio)

    if len(t1) >= 4 and len(t2) >= 4 and (t1 in t2 or t2 in t1):
        return 0.85

    seq_ratio = SequenceMatcher(None, t1, t2).ratio()
    return round(seq_ratio * 0.70, 2) if seq_ratio >= 0.52 else 0.0


def evaluate_color_match(lost_item, found_item, is_same_key, visual_colors1=None, visual_colors2=None):
    m1 = strip_accents(lost_item.mau_sac)
    m2 = strip_accents(found_item.mau_sac)
    has_m1 = m1 not in ["none", "null", ""]
    has_m2 = m2 not in ["none", "null", ""]

    is_multi1 = any(k in m1 for k in ["nhieu mau", "da sac", "du mau"])
    is_multi2 = any(k in m2 for k in ["nhieu mau", "da sac", "du mau"])

    text_colors1 = (
        extract_colors(lost_item.mau_sac)
        | extract_colors(lost_item.ten_do)
        | extract_colors(getattr(lost_item, "mo_ta", ""))
    )
    text_colors2 = (
        extract_colors(found_item.mau_sac)
        | extract_colors(found_item.ten_do)
        | extract_colors(getattr(found_item, "mo_ta", ""))
    )

    vis1 = set(visual_colors1) if visual_colors1 else set()
    vis2 = set(visual_colors2) if visual_colors2 else set()

    if is_multi1 or is_multi2:
        if is_multi1 and is_multi2:
            return 15, False
        if is_multi1 and (len(text_colors2) >= 2 or len(vis2) >= 2):
            return 15, False
        if is_multi2 and (len(text_colors1) >= 2 or len(vis1) >= 2):
            return 15, False
        return (11 if is_same_key else 4), False

    if text_colors1 and text_colors2:
        if text_colors1.intersection(text_colors2):
            return 15, False
        return 0, True

    if text_colors1 and not text_colors2:
        if vis2 and text_colors1.intersection(vis2):
            return (14 if is_same_key else 8), False
        return (10 if is_same_key else 4), False

    if text_colors2 and not text_colors1:
        if vis1 and text_colors2.intersection(vis1):
            return (14 if is_same_key else 8), False
        return (10 if is_same_key else 4), False

    if vis1 and vis2:
        if vis1.intersection(vis2):
            return (13 if is_same_key else 7), False
        return 3, False

    if has_m1 and has_m2:
        sim = SequenceMatcher(None, m1, m2).ratio()
        return int(sim * 15), (sim < 0.3)

    return (10 if is_same_key else 4), False


def location_score(loc1, loc2):
    v1 = strip_accents(normalize_location_text(loc1))
    v2 = strip_accents(normalize_location_text(loc2))
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


# ==============================================================================
# QUẢN LÝ MÃ PIN BỀN VỮNG TRONG DATABASE (NEON POSTGRESQL) + CONFIG FILE + ENV
# ==============================================================================
def get_current_pin():
    try:
        cfg = db.session.get(SystemConfig, "admin_pin")
        if cfg and cfg.value:
            return str(cfg.value).strip()
    except Exception:
        db.session.rollback()

    env_pin = os.environ.get("ADMIN_PIN", "").strip()
    if env_pin:
        return env_pin

    try:
        with open("config.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return str(data.get("admin_pin", "123456"))
    except Exception:
        return "123456"


def save_new_pin(new_pin):
    pin_str = str(new_pin).strip()
    saved_db = False
    try:
        cfg = db.session.get(SystemConfig, "admin_pin")
        if cfg:
            cfg.value = pin_str
        else:
            db.session.add(SystemConfig(key="admin_pin", value=pin_str))
        db.session.commit()
        saved_db = True
    except Exception as e:
        db.session.rollback()
        print(f"Lỗi lưu PIN vào DB: {e}")

    try:
        config_data = {}
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            config_data = {}
        config_data["admin_pin"] = pin_str
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception:
        return saved_db


def _score_pair_full(lost, found):
    """
    Hàm lõi chấm điểm 1 cặp (LostItem, FoundItem) theo 5 tiêu chí chuẩn 100 điểm:
    Hình ảnh (50) + Tên đồ (15) + Màu sắc (15) + Vị trí (10) + Thời gian (10) = 100.
    Dấu hiệu riêng được trích xuất làm thông tin đối chứng cho Giáo viên.
    """
    has1 = check_real_image(lost.hinh_anh)
    has2 = check_real_image(found.hinh_anh)

    key1 = get_core_keyword(lost.ten_do)
    key2 = get_core_keyword(found.ten_do)
    cat1 = auto_normalize_category(lost.ten_do, lost.loai_do)
    cat2 = auto_normalize_category(found.ten_do, found.loai_do)
    is_same_cat = (cat1 != "" and cat1 == cat2)
    ten_sim = similarity(lost.ten_do, found.ten_do)
    is_same_key = (ten_sim >= 0.72) or (key1 == key2 and key1 != "")

    if is_same_key:
        ten_score = 15 if ten_sim >= 0.85 else max(11, int(ten_sim * 15))
    else:
        ten_score = int(round(ten_sim * 15))

    v_cols1, v_cols2 = set(), set()
    if has1 and has2:
        for u in split_image_list(lost.hinh_anh):
            f1 = calculate_image_features(u)
            if f1 and f1.get("visual_colors"):
                v_cols1.update(f1.get("visual_colors"))
        for u in split_image_list(found.hinh_anh):
            f2 = calculate_image_features(u)
            if f2 and f2.get("visual_colors"):
                v_cols2.update(f2.get("visual_colors"))

    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key, v_cols1 or None, v_cols2 or None)
    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
    mark_hints = analyze_distinctive_marks(lost, found)

    if has1 and has2:
        anh_score, image_ratio, best_i1, best_i2, vis_same = image_score(
            lost.hinh_anh, found.hinh_anh,
            is_same_keyword=is_same_key,
            is_same_cat=is_same_cat,
            color_conflict=color_conflict
        )
    else:
        anh_score, image_ratio, best_i1, best_i2, vis_same = 0, 0.0, 0, 0, False

    if not is_same_key and vis_same and is_same_cat:
        ten_score = max(ten_score, 12)

    total_score = ten_score + mau_score + vi_tri_score + ngay_score + anh_score
    return {
        "score": total_score,
        "score_image": anh_score,
        "score_name": ten_score,
        "score_color": mau_score,
        "score_location": vi_tri_score,
        "score_date": ngay_score,
        "mark_hints": mark_hints,
        "color_conflict": color_conflict,
        "image_ratio": image_ratio,
        "best_lost_idx": best_i1,
        "best_found_idx": best_i2,
    }


@app.route("/api/analyze-image", methods=["POST"])
def api_analyze_image():
    """API nhận ảnh upload từ form và trả về gợi ý Màu sắc + Loại đồ bằng MobileNetV2 & HSV."""
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"ok": False, "error": "Chưa chọn ảnh"})

    try:
        raw_bytes = file.read(2 * 1024 * 1024)
        with Image.open(io.BytesIO(raw_bytes)) as src:
            image = ImageOps.exif_transpose(src).convert("RGB")
            _, small_48, fg_mask = _extract_foreground_pil(image)
            visual_colors = _detect_visual_colors_masked(small_48, fg_mask)
            _, top_ids = _extract_mobilenet_embedding(image)

        display_colors = [
            c for c in ["đen", "trắng", "xám", "đỏ", "xanh", "vàng", "hồng", "tím", "cam", "nâu"]
            if c in visual_colors
        ]
        color_str = ", ".join(display_colors[:3]) if display_colors else ""
        suggested_cat = _predict_category_from_ids(top_ids)

        return jsonify({
            "ok": True,
            "colors": color_str,
            "category": suggested_cat
        })
    except Exception as e:
        print(f"Lỗi AI Auto-Fill: {e}")
        return jsonify({"ok": False, "error": str(e)})


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/lost", methods=["GET", "POST"])
def lost():
    if request.method == "POST":
        ten_do = request.form.get("ten_do", "").strip()
        loai_do = auto_normalize_category(ten_do, request.form.get("loai_do", ""))
        mau_sac = request.form.get("mau_sac", "").strip()
        noi_mat = normalize_location_text(request.form.get("noi_mat", ""))
        ngay_mat = request.form.get("ngay_mat", "").strip()
        mo_ta = clean_note_text(request.form.get("mo_ta", ""))

        existing_dup = LostItem.query.filter_by(
            ten_do=ten_do, noi_mat=noi_mat, ngay_mat=ngay_mat, trang_thai="DANG_TIM"
        ).order_by(LostItem.id.desc()).first()
        if existing_dup and strip_accents(existing_dup.mo_ta) == strip_accents(mo_ta):
            return redirect(url_for("match_items", item_id=existing_dup.id))

        files = request.files.getlist("hinh_anh")
        filename = upload_multiple_images(files, max_images=3)

        new_item_id = None
        best_match_score = 0
        try:
            item = LostItem(
                ten_do=ten_do,
                loai_do=loai_do,
                mau_sac=mau_sac,
                noi_mat=noi_mat,
                ngay_mat=ngay_mat,
                mo_ta=mo_ta,
                hinh_anh=filename,
                trang_thai="DANG_TIM"
            )
            db.session.add(item)
            db.session.commit()
            new_item_id = item.id

            active_found = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA").order_by(FoundItem.id.desc()).limit(25).all()
            for f_item in active_found:
                res = _score_pair_full(item, f_item)
                if res["score"] > best_match_score:
                    best_match_score = res["score"]
            _persist_disk_cache()
        except Exception as db_err:
            print("LOI LUU LOST:", db_err)
            try:
                db.session.rollback()
            except Exception:
                pass

        if new_item_id:
            return redirect(url_for("match_items", item_id=new_item_id, instant_score=best_match_score))
        return redirect(url_for("list_items"))
    return render_template("lost_form.html")


@app.route("/found", methods=["GET", "POST"])
def found():
    if request.method == "POST":
        ten_do = request.form.get("ten_do", "").strip()
        loai_do = auto_normalize_category(ten_do, request.form.get("loai_do", ""))
        mau_sac = request.form.get("mau_sac", "").strip()
        noi_nhat = normalize_location_text(request.form.get("noi_nhat", ""))
        ngay_nhat = request.form.get("ngay_nhat", "").strip()
        noi_luu_giu = normalize_location_text(request.form.get("noi_luu_giu", ""))
        mo_ta = clean_note_text(request.form.get("mo_ta", ""))

        existing_dup = FoundItem.query.filter_by(
            ten_do=ten_do, noi_nhat=noi_nhat, ngay_nhat=ngay_nhat, trang_thai="DANG_LUU_GIU"
        ).order_by(FoundItem.id.desc()).first()
        if existing_dup and strip_accents(existing_dup.mo_ta) == strip_accents(mo_ta):
            return redirect(url_for("match_items", found_id=existing_dup.id))

        files = request.files.getlist("hinh_anh")
        filename = upload_multiple_images(files, max_images=3)

        new_found_id = None
        best_match_score = 0
        try:
            item = FoundItem(
                ten_do=ten_do,
                loai_do=loai_do,
                mau_sac=mau_sac,
                noi_nhat=noi_nhat,
                ngay_nhat=ngay_nhat,
                noi_luu_giu=noi_luu_giu,
                mo_ta=mo_ta,
                hinh_anh=filename,
                trang_thai="DANG_LUU_GIU"
            )
            db.session.add(item)
            db.session.commit()
            new_found_id = item.id

            active_lost = LostItem.query.filter(LostItem.trang_thai != "DA_TIM").order_by(LostItem.id.desc()).limit(25).all()
            for l_item in active_lost:
                res = _score_pair_full(l_item, item)
                if res["score"] > best_match_score:
                    best_match_score = res["score"]
            _persist_disk_cache()
        except Exception as db_err:
            print("LOI LUU FOUND:", db_err)
            try:
                db.session.rollback()
            except Exception:
                pass

        if new_found_id:
            return redirect(url_for("match_items", found_id=new_found_id, instant_score=best_match_score))
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
            lost_query = lost_query.filter(
                (LostItem.ten_do.ilike(f"%{keyword}%")) | (LostItem.mo_ta.ilike(f"%{keyword}%"))
            )
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
            found_query = found_query.filter(
                (FoundItem.ten_do.ilike(f"%{keyword}%")) | (FoundItem.mo_ta.ilike(f"%{keyword}%"))
            )
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


@app.route("/delete-lost/<int:item_id>", methods=["GET", "POST"])
@app.route("/delete/<int:item_id>", methods=["GET", "POST"])
def delete_item(item_id):
    pin_code = (
        request.form.get("pin_code", "").strip()
        if request.method == "POST"
        else request.args.get("pin_code", "").strip()
    )
    if pin_code != get_current_pin():
        return """
        <script>
            alert("Mã PIN Giáo viên không chính xác! Không thể xóa bài đăng.");
            window.history.back();
        </script>
        """
    try:
        item = LostItem.query.get_or_404(item_id)
        cleanup_local_images(item.hinh_anh)
        db.session.delete(item)
        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
    return redirect(url_for("list_items"))


@app.route("/delete-found/<int:item_id>", methods=["GET", "POST"])
def delete_found(item_id):
    pin_code = (
        request.form.get("pin_code", "").strip()
        if request.method == "POST"
        else request.args.get("pin_code", "").strip()
    )
    if pin_code != get_current_pin():
        return """
        <script>
            alert("Mã PIN Giáo viên không chính xác! Không thể xóa bài đăng.");
            window.history.back();
        </script>
        """
    try:
        item = FoundItem.query.get_or_404(item_id)
        cleanup_local_images(item.hinh_anh)
        db.session.delete(item)
        db.session.commit()
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
    return redirect(url_for("found_list"))


@app.route("/edit-lost/<int:item_id>", methods=["GET", "POST"])
@app.route("/edit/<int:item_id>", methods=["GET", "POST"])
def edit_lost(item_id):
    try:
        item = LostItem.query.get_or_404(item_id)
    except Exception:
        return redirect(url_for("list_items"))

    if request.method == "POST":
        old_all = split_image_list(item.hinh_anh)

        item.ten_do = request.form.get("ten_do")
        item.loai_do = auto_normalize_category(item.ten_do, request.form.get("loai_do"))
        item.mau_sac = request.form.get("mau_sac")
        item.noi_mat = normalize_location_text(request.form.get("noi_mat", ""))
        item.ngay_mat = request.form.get("ngay_mat")
        item.mo_ta = clean_note_text(request.form.get("mo_ta", ""))

        kept_old = split_image_list(request.form.get("existing_images", item.hinh_anh))
        files = request.files.getlist("hinh_anh")
        if any(f and f.filename != "" for f in files):
            new_imgs = upload_multiple_images(files, max_images=3)
            new_list = split_image_list(new_imgs)
            merged = (kept_old + new_list)[:3]
            item.hinh_anh = ",".join(merged)
        else:
            item.hinh_anh = ",".join(kept_old)

        new_all = set(split_image_list(item.hinh_anh))
        removed_imgs = [u for u in old_all if u not in new_all]
        if removed_imgs:
            cleanup_local_images(",".join(removed_imgs))

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
        old_all = split_image_list(item.hinh_anh)

        item.ten_do = request.form.get("ten_do")
        item.loai_do = auto_normalize_category(item.ten_do, request.form.get("loai_do"))
        item.mau_sac = request.form.get("mau_sac")
        item.noi_nhat = normalize_location_text(request.form.get("noi_nhat", ""))
        item.ngay_nhat = request.form.get("ngay_nhat")
        item.noi_luu_giu = normalize_location_text(request.form.get("noi_luu_giu", ""))
        item.mo_ta = clean_note_text(request.form.get("mo_ta", ""))

        kept_old = split_image_list(request.form.get("existing_images", item.hinh_anh))
        files = request.files.getlist("hinh_anh")
        if any(f and f.filename != "" for f in files):
            new_imgs = upload_multiple_images(files, max_images=3)
            new_list = split_image_list(new_imgs)
            merged = (kept_old + new_list)[:3]
            item.hinh_anh = ",".join(merged)
        else:
            item.hinh_anh = ",".join(kept_old)

        new_all = set(split_image_list(item.hinh_anh))
        removed_imgs = [u for u in old_all if u not in new_all]
        if removed_imgs:
            cleanup_local_images(",".join(removed_imgs))

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
        tong_found = FoundItem.query.count()
        all_items = LostItem.query.all()
        all_found = FoundItem.query.all()
    except Exception:
        tong_so, tong_found, all_items, all_found = 0, 0, [], []

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

    da_tra_found = [f for f in all_found if f.trang_thai in ("DA_TRA", "Đã hoàn trả")]
    da_tim_lost = [l for l in all_items if l.trang_thai in ("DA_TIM", "Đã tìm thấy")]

    so_da_hoan_tra = len(da_tra_found)
    ty_le_hoan_tra = round((so_da_hoan_tra / tong_found) * 100, 1) if tong_found > 0 else 0.0

    durations_hours = []
    for f in da_tra_found:
        if getattr(f, "thoi_diem_tra", None) and f.ngay_nhat:
            dt_nhat = parse_date_flexible(f.ngay_nhat)
            if dt_nhat:
                diff_sec = (f.thoi_diem_tra - dt_nhat).total_seconds()
                if diff_sec >= 0:
                    durations_hours.append(diff_sec / 3600.0)

    if durations_hours:
        avg_hours = sum(durations_hours) / len(durations_hours)
        if avg_hours < 24:
            thoi_gian_tb = f"{round(avg_hours, 1)} giờ"
        else:
            thoi_gian_tb = f"{round(avg_hours / 24.0, 1)} ngày"
    else:
        thoi_gian_tb = "Chưa đủ mẫu"

    return render_template(
        "dashboard.html",
        tong_so=tong_so,
        tong_found=tong_found,
        so_da_hoan_tra=so_da_hoan_tra,
        so_da_tim=len(da_tim_lost),
        ty_le_hoan_tra=ty_le_hoan_tra,
        thoi_gian_tb=thoi_gian_tb,
        thong_ke=thong_ke
    )


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


@app.route("/found-return/<int:id>", methods=["GET", "POST"])
def found_return(id):
    pin_code = (
        request.form.get("pin_code", "").strip()
        if request.method == "POST"
        else request.args.get("pin_code", "").strip()
    )
    current_pin = get_current_pin()

    if not pin_code:
        return f"""
        <script>
            var p = prompt("Nhập Mã PIN Giáo viên để xác nhận đã trả đồ:");
            if (p !== null && p.trim() !== "") {{
                window.location.href = "/found-return/{id}?pin_code=" + encodeURIComponent(p.trim());
            }} else {{
                window.history.back();
            }}
        </script>
        """

    if pin_code != current_pin:
        return """
        <script>
            alert("Mã PIN Giáo viên không chính xác! Không thể xác nhận trả đồ.");
            window.history.back();
        </script>
        """

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
    """
    Đối sánh v58: Thang 5 tiêu chí chuẩn 100 điểm (50 + 15 + 15 + 10 + 10 = 100).
    Hỗ trợ lọc riêng theo từng đồ báo mất (item_id) hoặc từng đồ nhặt được (found_id).
    """
    page = request.args.get("page", 1, type=int)
    cat_filter = request.args.get("category", "").strip()
    found_id = request.args.get("found_id", type=int)
    instant_score = request.args.get("instant_score", 0, type=int)
    per_page = 10

    try:
        lost_query = LostItem.query.filter(LostItem.trang_thai != "DA_TIM")
        found_query = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA")

        if item_id:
            lost_query = lost_query.filter(LostItem.id == item_id)
        if found_id:
            found_query = found_query.filter(FoundItem.id == found_id)
        if cat_filter:
            lost_query = lost_query.filter(LostItem.loai_do == cat_filter)
            found_query = found_query.filter(FoundItem.loai_do == cat_filter)

        lost_items = lost_query.order_by(LostItem.id.desc()).all()
        found_items = found_query.order_by(FoundItem.id.desc()).all()
        total_combinations = len(lost_items) * len(found_items)

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
            cat1 = auto_normalize_category(lost.ten_do, lost.loai_do)
            for found in found_items:
                try:
                    key2 = get_core_keyword(found.ten_do)
                    cat2 = auto_normalize_category(found.ten_do, found.loai_do)
                    ten_sim = similarity(lost.ten_do, found.ten_do)
                    is_same_key = (ten_sim >= 0.72) or (key1 == key2 and key1 != "")
                    is_same_cat = (cat1 != "" and cat1 == cat2)

                    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key)
                    mark_hints = analyze_distinctive_marks(lost, found)
                    both_have_imgs = lost.has_image and found.has_image

                    if not is_same_key and not mark_hints:
                        if not is_same_cat or not both_have_imgs or color_conflict:
                            continue

                    sig = (
                        strip_accents(lost.ten_do),
                        strip_accents(lost.noi_mat),
                        str(lost.ngay_mat or "").strip(),
                        strip_accents(lost.mo_ta),
                        strip_accents(found.ten_do),
                        strip_accents(found.noi_nhat),
                        str(found.ngay_nhat or "").strip(),
                        strip_accents(found.mo_ta),
                    )
                    if sig in seen_signatures:
                        continue
                    seen_signatures.add(sig)

                    if is_same_key:
                        ten_score = 15 if ten_sim >= 0.85 else max(11, int(ten_sim * 15))
                    else:
                        ten_score = int(round(ten_sim * 15))

                    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
                    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
                    base_score = ten_score + mau_score + vi_tri_score + ngay_score

                    has_both_images = 1 if both_have_imgs else 0
                    non_conflict = 0 if color_conflict else 1
                    same_key_flag = 1 if is_same_key else 0
                    has_mark_hint = 1 if mark_hints else 0

                    sequence += 1
                    candidates.append({
                        "lost": lost,
                        "found": found,
                        "base_score": base_score,
                        "same_key_flag": same_key_flag,
                        "has_mark_hint": has_mark_hint,
                        "has_both_images": has_both_images,
                        "non_conflict": non_conflict,
                        "is_same_key": is_same_key,
                        "is_same_cat": is_same_cat,
                        "color_conflict": color_conflict,
                        "score_name": ten_score,
                        "score_color": mau_score,
                        "score_location": vi_tri_score,
                        "score_date": ngay_score,
                        "mark_hints": mark_hints,
                        "seq": sequence
                    })
                except Exception:
                    continue

        candidates.sort(
            key=lambda x: (x["same_key_flag"], x["non_conflict"], x["has_mark_hint"], x["has_both_images"], x["base_score"], x["seq"]),
            reverse=True
        )

        urls_to_fetch = set()
        for item in candidates[:30]:
            if item["lost"].has_image and item["found"].has_image:
                for u in split_image_list(item["lost"].hinh_anh):
                    cached_f = get_feature_cache(u)
                    if cached_f is None or cached_f.get("version") != CACHE_VERSION:
                        urls_to_fetch.add(u)
                for u in split_image_list(item["found"].hinh_anh):
                    cached_f = get_feature_cache(u)
                    if cached_f is None or cached_f.get("version") != CACHE_VERSION:
                        urls_to_fetch.add(u)

        if urls_to_fetch:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(calculate_image_features, urls_to_fetch))
            _persist_disk_cache()

        scored_matches = []
        for item in candidates:
            lost, found = item["lost"], item["found"]
            v_cols1, v_cols2 = set(), set()
            if lost.has_image and found.has_image:
                for u in split_image_list(lost.hinh_anh):
                    f1 = get_feature_cache(u)
                    if f1 and f1.get("visual_colors"):
                        v_cols1.update(f1.get("visual_colors"))
                for u in split_image_list(found.hinh_anh):
                    f2 = get_feature_cache(u)
                    if f2 and f2.get("visual_colors"):
                        v_cols2.update(f2.get("visual_colors"))

            mau_score, color_conflict = evaluate_color_match(
                lost, found, item["is_same_key"], v_cols1 or None, v_cols2 or None
            )

            if lost.has_image and found.has_image:
                anh_score, _, best_i1, best_i2, vis_same = image_score(
                    lost.hinh_anh, found.hinh_anh,
                    is_same_keyword=item["is_same_key"],
                    is_same_cat=item["is_same_cat"],
                    color_conflict=color_conflict
                )
            else:
                anh_score, best_i1, best_i2, vis_same = 0, 0, 0, False

            ten_score_final = item["score_name"]
            if not item["is_same_key"] and vis_same and item["is_same_cat"]:
                ten_score_final = max(ten_score_final, 12)

            total_score = (
                anh_score
                + ten_score_final
                + mau_score
                + item["score_location"]
                + item["score_date"]
            )

            if not item["is_same_key"] and not item["mark_hints"] and total_score < 55:
                continue

            if total_score >= 75:
                ket_luan = "🟢 Khả năng trùng khớp cao"
            elif total_score >= 55:
                ket_luan = "🟡 Có khả năng trùng khớp"
            elif total_score >= 38:
                ket_luan = "🟠 Có một số điểm tương đồng"
            else:
                ket_luan = "🔴 Khả năng trùng khớp thấp"

            scored_matches.append({
                "lost": lost,
                "found": found,
                "score": total_score,
                "ket_luan": ket_luan,
                "score_image": anh_score,
                "score_name": ten_score_final,
                "score_color": mau_score,
                "score_location": item["score_location"],
                "score_date": item["score_date"],
                "mark_hints": item["mark_hints"],
                "best_lost_idx": best_i1,
                "best_found_idx": best_i2,
            })

        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        total_matches = len(scored_matches)
        filtered_out = max(0, total_combinations - total_matches)
        total_pages = max(1, (total_matches + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        page_matches = scored_matches[start_idx:start_idx + per_page]
        gc.collect()

        return render_template(
            "match.html",
            matches=page_matches,
            page=page,
            total_pages=total_pages,
            total_matches=total_matches,
            total_combinations=total_combinations,
            filtered_out=filtered_out,
            instant_score=instant_score,
            selected_category=cat_filter,
            item_id=item_id,
            found_id=found_id
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
            total_combinations=0,
            filtered_out=0,
            instant_score=0,
            selected_category=cat_filter,
            item_id=item_id,
            found_id=found_id
        )


@app.route("/teacher-check/<int:lost_id>/<int:found_id>", methods=["GET", "POST"])
def teacher_check(lost_id, found_id):
    if request.method == "POST":
        action = request.form.get("action")
        if action == "unlock_pin":
            pin_code = request.form.get("pin_code", "").strip()
            if pin_code == get_current_pin():
                session["teacher_unlocked"] = True
            else:
                return """
                <script>
                    alert("Mã PIN Giáo viên không chính xác! Không thể mở khóa thông tin gốc.");
                    window.history.back();
                </script>
                """
        elif action == "lock_pin":
            session.pop("teacher_unlocked", None)

    is_unlocked = bool(session.get("teacher_unlocked", False))

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

    res = _score_pair_full(lost, found)
    _persist_disk_cache()

    score = res["score"]
    base_score = score - res["score_image"]

    if score >= 75:
        ket_luan = "🟢 Khả năng trùng khớp cao"
    elif score >= 55:
        ket_luan = "🟡 Có khả năng trùng khớp"
    elif score >= 38:
        ket_luan = "🟠 Có một số điểm tương đồng"
    else:
        ket_luan = "🔴 Khả năng trùng khớp thấp"

    image_warning = None
    if res["color_conflict"]:
        image_warning = "⚠️ Hai món đồ có màu sắc trái ngược nhau, giáo viên cần đối chiếu kỹ."
    elif has1 and has2 and res["image_ratio"] < 0.25:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."
    elif not (has1 and has2):
        image_warning = "⚠️ Bài đăng này thiếu hình ảnh minh họa, vui lòng xác minh bằng thông tin mô tả."

    gc.collect()
    return render_template(
        "teacher_check.html",
        lost=lost,
        found=found,
        score=score,
        base_score=base_score,
        color_score=res["score_color"],
        server_img_score=res["score_image"],
        mark_hints=res["mark_hints"],
        ket_luan=ket_luan,
        image_warning=image_warning,
        is_unlocked=is_unlocked
    )


@app.route("/confirm-return/<int:lost_id>/<int:found_id>", methods=["GET", "POST"])
def confirm_return(lost_id, found_id):
    pin_code = request.form.get("pin_code", "").strip() if request.method == "POST" else request.args.get("pin_code", "").strip()
    current_pin = get_current_pin()

    if not session.get("teacher_unlocked") and pin_code != current_pin:
        return """
        <script>
            alert("Mã PIN Giáo viên không chính xác! Vui lòng mở khóa trước khi xác nhận hoàn trả.");
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

        lost_items = lost_query.all()
        found_items = found_query.all()

        time_label = "Toan_Thoi_Gian"
        if year or month:
            time_label = (
                f"Thang_{int(month):02d}_{year}" if (month and year)
                else (f"Thang_{int(month):02d}" if month else f"Nam_{year}")
            )
            filtered_lost = []
            for item in lost_items:
                dt = parse_date_flexible(item.ngay_mat)
                if dt:
                    if year and str(dt.year) != str(year):
                        continue
                    if month and dt.month != int(month):
                        continue
                    filtered_lost.append(item)
            lost_items = filtered_lost

            filtered_found = []
            for item in found_items:
                dt = parse_date_flexible(item.ngay_nhat)
                if dt:
                    if year and str(dt.year) != str(year):
                        continue
                    if month and dt.month != int(month):
                        continue
                    filtered_found.append(item)
            found_items = filtered_found

        lost_data = [{
            "STT": i + 1,
            "Tên đồ vật": item.ten_do,
            "Loại đồ": item.loai_do,
            "Màu sắc": item.mau_sac,
            "Dấu hiệu riêng / Mô tả": clean_note_text(getattr(item, "mo_ta", "")),
            "Nơi mất": normalize_location_text(item.noi_mat),
            "Ngày mất": format_vn_date(item.ngay_mat),
            "Trạng thái": "Đã tìm thấy" if item.trang_thai in ["DA_TIM", "Đã tìm thấy"] else "Đang tìm"
        } for i, item in enumerate(lost_items)]

        found_data = [{
            "STT": i + 1,
            "Tên đồ vật": item.ten_do,
            "Loại đồ": item.loai_do,
            "Màu sắc": item.mau_sac,
            "Dấu hiệu riêng / Mô tả": clean_note_text(getattr(item, "mo_ta", "")),
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


@app.route("/verify-pin", methods=["POST"])
def verify_pin():
    data = request.get_json(silent=True) or {}
    pin_input = str(data.get("pin", request.form.get("pin", ""))).strip()
    return jsonify({"valid": pin_input == get_current_pin()})


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
    with app.app_context():
        db.create_all()
    app.run(debug=True)
