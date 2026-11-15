import os
import io
import uuid
import json
import re
import gc
import base64
from collections import OrderedDict
from sqlalchemy import text, inspect
import urllib3
from datetime import datetime, timezone, timedelta
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader

from PIL import Image, ImageOps, ImageFile

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

VN_TZ = timezone(timedelta(hours=7))

HAS_OPENCV = False
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except Exception as e:
    print(f"Cảnh báo OpenCV không khả dụng: {e}")

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

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Giới hạn tổng dung lượng request để bảo vệ RAM/CPU trên Render Free.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

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
    except Exception as e:
        print(f"Lỗi Create DB All: {e}")

    # Tự bổ sung cột đặc trưng cho database đang tồn tại, không xóa dữ liệu cũ.
    try:
        inspector = inspect(db.engine)
        for table_name in ("lost_item", "found_item"):
            if table_name in inspector.get_table_names():
                cols = {c["name"] for c in inspector.get_columns(table_name)}
                if "image_features" not in cols:
                    db.session.execute(text(f"ALTER TABLE {table_name} ADD COLUMN image_features TEXT"))
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.warning("Không thể kiểm tra/nâng cấp cột image_features: %s", e)

from collections import OrderedDict
import hashlib
import requests

# Cache LRU: không xóa toàn bộ cache khi thêm ảnh mới.
FEATURE_CACHE = OrderedDict()
MAX_CACHE_SIZE = 256
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
IMAGE_DOWNLOAD_TIMEOUT = (3, 6)
_IMAGE_HTTP = requests.Session()

STOP_WORDS_VN = {"con", "cái", "chiếc", "hộp", "bộ", "cặp", "đôi", "quả", "trái", "tấm", "bức", "tờ", "cục"}


def _image_cache_key(img_path):
    return str(img_path).strip() if img_path else None


def check_real_image(img_path):
    if not img_path:
        return False
    try:
        value = str(img_path).strip()
        lowered = value.lower()
        if lowered in {"", "none", "null", "undefined"}:
            return False
        invalid = ("chua co anh", "chưa có ảnh", "no_image", "placeholder", "120?text=")
        if any(word in lowered for word in invalid):
            return False
        if lowered.startswith(("http://", "https://")):
            return True  # Kiểm tra nội dung thật khi tải, tránh request thừa ở vòng ghép cặp.
        path = value if os.path.isabs(value) else os.path.join(
            app.config.get("UPLOAD_FOLDER", "static/uploads"), value
        )
        return os.path.isfile(path)
    except (TypeError, OSError):
        return False


def get_feature_cache(img_path):
    key = _image_cache_key(img_path)
    if not key:
        return None
    value = FEATURE_CACHE.get(key)
    if value is not None:
        FEATURE_CACHE.move_to_end(key)
    return value


def save_feature_cache(img_path, features):
    key = _image_cache_key(img_path)
    if not key or features is None:
        return
    FEATURE_CACHE[key] = features
    FEATURE_CACHE.move_to_end(key)
    while len(FEATURE_CACHE) > MAX_CACHE_SIZE:
        FEATURE_CACHE.popitem(last=False)


def _read_image_bytes(img_path):
    value = str(img_path).strip()
    if value.lower().startswith(("http://", "https://")):
        response = _IMAGE_HTTP.get(
            value, timeout=IMAGE_DOWNLOAD_TIMEOUT, stream=True,
            headers={"User-Agent": "SmartLostFound/1.0"}
        )
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise ValueError("Ảnh vượt quá giới hạn dung lượng xử lý")
            chunks.append(chunk)
        response.close()
        return b"".join(chunks)

    path = value if os.path.isabs(value) else os.path.join(app.config["UPLOAD_FOLDER"], value)
    with open(path, "rb") as file:
        data = file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn dung lượng xử lý")
    return data


def calculate_image_features(img_path):
    if not check_real_image(img_path):
        return None

    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        data = _read_image_bytes(img_path)
        if not data:
            return None

        # Đọc ảnh hoàn chỉnh, tự xoay theo EXIF và giữ nguyên tỷ lệ.
        with Image.open(io.BytesIO(data)) as original:
            # Chặn ảnh có số pixel quá lớn trước khi giải mã/chuyển RGB.
            width, height = original.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Độ phân giải ảnh vượt giới hạn xử lý")
            original = ImageOps.exif_transpose(original).convert("RGB")
            # Chỉ resize bản dùng để trích xuất đặc trưng; ảnh gốc trên Cloudinary không đổi.
            original.thumbnail((320, 320), Image.Resampling.LANCZOS)
            rgb = np.asarray(original) if HAS_OPENCV else None
            gray_pil = original.convert("L").resize((32, 32), Image.Resampling.LANCZOS)

            # Perceptual hash (DCT pHash): ổn định hơn dHash khi ảnh thay đổi kích thước.
            if HAS_OPENCV:
                pixels = np.asarray(gray_pil, dtype=np.float32)
                dct = cv2.dct(pixels)
                block = dct[:8, :8]
                median = float(np.median(block.flatten()[1:]))
                phash = 0
                for value in block.flatten():
                    phash = (phash << 1) | int(value > median)
            else:
                small = gray_pil.resize((9, 8), Image.Resampling.LANCZOS)
                vals = list(small.getdata())
                dhash = 0
                for row in range(8):
                    for col in range(8):
                        dhash = (dhash << 1) | int(vals[row * 9 + col] > vals[row * 9 + col + 1])
                phash = dhash

            features = {"phash": phash, "orb": None, "hsv_hist": None}

            if HAS_OPENCV and rgb is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                orb = cv2.ORB_create(
                    nfeatures=500, scaleFactor=1.2, nlevels=8,
                    edgeThreshold=15, fastThreshold=12
                )
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                features["orb"] = descriptors
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
                features["hsv_hist"] = hist

            save_feature_cache(img_path, features)
            return features
    except Exception as exc:
        app.logger.warning("Không xử lý được ảnh %s: %s", img_path, exc)
        return None


def serialize_image_features(features):
    """Chuyển đặc trưng ảnh thành JSON nhỏ, lưu bền vững trong database."""
    if not features:
        return None
    try:
        orb = features.get("orb")
        hist = features.get("hsv_hist")
        payload = {
            "v": 1,
            "phash": int(features.get("phash", 0)),
            "orb": orb.tolist() if orb is not None else None,
            "hsv_hist": hist.reshape(-1).astype(float).tolist() if hist is not None else None,
        }
        return json.dumps(payload, separators=(",", ":"))
    except Exception:
        return None


def deserialize_image_features(value):
    if not value:
        return None
    try:
        payload = json.loads(value) if isinstance(value, str) else value
        result = {"phash": int(payload["phash"]), "orb": None, "hsv_hist": None}
        if HAS_OPENCV and payload.get("orb"):
            result["orb"] = np.asarray(payload["orb"], dtype=np.uint8)
        if HAS_OPENCV and payload.get("hsv_hist"):
            result["hsv_hist"] = np.asarray(payload["hsv_hist"], dtype=np.float32).reshape(12, 8)
        return result
    except Exception:
        return None


def get_item_features(item):
    """Ưu tiên đặc trưng trong DB; chỉ xử lý ảnh khi bản ghi cũ chưa có."""
    stored = deserialize_image_features(getattr(item, "image_features", None))
    if stored is not None:
        return stored
    extracted = calculate_image_features(item.hinh_anh)
    if extracted:
        serialized = serialize_image_features(extracted)
        if serialized:
            item.image_features = serialized
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
    return extracted


def image_score_from_features(f1, f2):
    """Chấm điểm ảnh trên thang 50; xử lý đúng pHash 64-bit và ORB đối sánh chéo."""
    if not f1 or not f2:
        return 0, 0.0

    # pHash được tạo từ khối DCT 8x8 nên có 64 bit, không phải 32 bit.
    hash_similarity = 0.0
    hash1, hash2 = f1.get("phash"), f2.get("phash")
    if hash1 is not None and hash2 is not None:
        distance = (int(hash1) ^ int(hash2)).bit_count()
        hash_similarity = max(0.0, 1.0 - distance / 64.0)

    # ORB: dùng crossCheck để tránh Lowe ratio test loại nhầm các descriptor
    # trùng khớp hoàn toàn (trường hợp hai ảnh giống hệt nhau).
    orb_similarity = 0.0
    des1, des2 = f1.get("orb"), f2.get("orb")
    if HAS_OPENCV and des1 is not None and des2 is not None and len(des1) and len(des2):
        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            good = sum(1 for match in matches if match.distance <= 48)
            denominator = max(1, min(len(des1), len(des2)))
            orb_similarity = max(0.0, min(1.0, good / denominator))
        except Exception:
            orb_similarity = 0.0

    color_similarity = 0.0
    h1, h2 = f1.get("hsv_hist"), f2.get("hsv_hist")
    if HAS_OPENCV and h1 is not None and h2 is not None:
        try:
            distance = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
            color_similarity = max(0.0, min(1.0, 1.0 - distance))
        except Exception:
            color_similarity = 0.0

    # Nếu ORB không thể trích xuất (ảnh quá đơn giản), phân bổ phần trọng số
    # đó sang pHash thay vì mặc định khiến ảnh giống nhau bị chặn ở 22/50.
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        score = hash_similarity * 45 + color_similarity * 5
    else:
        score = orb_similarity * 28 + hash_similarity * 17 + color_similarity * 5

    score = int(round(max(0.0, min(50.0, score))))
    return score, score / 50.0


def image_score(img1_filename, img2_filename, is_same_keyword=False):
    if not img1_filename or not img2_filename:
        return 0, 0.0
    if str(img1_filename).strip() == str(img2_filename).strip():
        return 50, 1.0
    return image_score_from_features(
        calculate_image_features(img1_filename),
        calculate_image_features(img2_filename)
    )


def get_core_keyword(text_name):
    if not text_name:
        return ""
    t = str(text_name).lower().strip()
    
    synonyms_map = {
        "máy tính casio": ["casio", "fx580", "fx 580", "máy tính bỏ túi", "casio580"],
        "laptop": ["laptop", "macbook", "máy tính xách tay"],
        "chuột": ["chuột", "con chuột", "chuột máy tính", "mouse"],
        "sách toán 10": ["sách toán 10", "toán 10"],
        "sách toán 9": ["sách toán 9", "toán 9", "sách bài tập toán 9", "sbt toán 9"],
        "sách văn": ["sách văn", "ngữ văn", "văn học"],
        "sách tiếng anh": ["sách anh", "tiếng anh", "english"],
        "sách tin học": ["tin học", "sách tin", "tin 11", "tin 9", "tin 8", "sách tin 8"],
        "sách công nghệ": ["công nghệ", "sách công nghệ", "công nghệ 8"],
        "sách lý": ["vật lý", "lý"],
        "sách hóa": ["hóa học", "hóa"],
        "bút bi": ["bút bi", "viết bi"],
        "bút chì": ["bút chì", "bút chì kim"],
        "điện thoại": ["điện thoại", "phone", "iphone", "samsung"],
        "balo": ["balo", "ba lô"],
        "cặp": ["cặp", "cặp sách"],
        "ví": ["ví", "bóp", "ví tiền"],
        "áo khoác": ["áo khoác", "áo gió", "áo ấm"],
        "nón": ["nón", "mũ", "nón kết"],
        "thẻ": ["thẻ", "căn cước", "cccd", "thẻ học sinh"],
        "bình nước": ["bình nước", "bình giữ nhiệt"]
    }

    for core_key, words in synonyms_map.items():
        if any(w in t for w in words):
            return core_key
            
    tokens = [w for w in t.split() if w not in STOP_WORDS_VN]
    return tokens[0] if tokens else t

def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()
    
    if t1 == t2:
        return 1.0

    nums1 = re.findall(r'\d+', t1)
    nums2 = re.findall(r'\d+', t2)
    if nums1 and nums2 and nums1 != nums2:
        return 0.1

    k1 = get_core_keyword(t1)
    k2 = get_core_keyword(t2)

    if k1 == k2 and k1 != "":
        return 0.90

    if t1 in t2 or t2 in t1:
        return 0.85

    return SequenceMatcher(None, t1, t2).ratio()

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
        
        try:
            item = LostItem(
                ten_do=request.form.get("ten_do", ""),
                loai_do=request.form.get("loai_do", ""),
                mau_sac=request.form.get("mau_sac", ""),
                noi_mat=request.form.get("noi_mat", ""),
                ngay_mat=request.form.get("ngay_mat", ""),
                hinh_anh=filename,
                trang_thai="DANG_TIM"
            )
            db.session.add(item)
            db.session.commit()
            if item.hinh_anh:
                features = calculate_image_features(item.hinh_anh)
                item.image_features = serialize_image_features(features)
                db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("Lỗi lưu đồ thất lạc")
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

        try:
            item = FoundItem(
                ten_do=request.form.get("ten_do", ""),
                loai_do=request.form.get("loai_do", ""),
                mau_sac=request.form.get("mau_sac", ""),
                noi_nhat=request.form.get("noi_nhat", ""),
                ngay_nhat=request.form.get("ngay_nhat", ""),
                hinh_anh=filename,
                trang_thai="DANG_LUU_GIU"
            )
            db.session.add(item)
            db.session.commit()
            if item.hinh_anh:
                features = calculate_image_features(item.hinh_anh)
                item.image_features = serialize_image_features(features)
                db.session.commit()
        except Exception:
            db.session.rollback()
            app.logger.exception("Lỗi lưu đồ nhặt được")
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
    except Exception:
        items = []
    return render_template("list_items.html", items=items)

@app.route("/found-list")
def found_list():
    try:
        items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    except Exception:
        items = []
    return render_template("found_list.html", items=items)

@app.route("/search")
def search():
    keyword = request.args.get("keyword", "").strip()
    category = request.args.get("category", "").strip()
    location = request.args.get("location", "").strip()
    color = request.args.get("color", "").strip()
    status = request.args.get("status", "").strip()

    try:
        lost_query = LostItem.query
        if keyword:
            lost_query = lost_query.filter(LostItem.ten_do.ilike(f"%{keyword}%"))
        if category:
            lost_query = lost_query.filter(LostItem.loai_do == category)
        if location:
            lost_query = lost_query.filter(LostItem.noi_mat == location)
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
            found_query = found_query.filter(FoundItem.noi_nhat == location)
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
        item.noi_mat = request.form.get("noi_mat")
        item.ngay_mat = request.form.get("ngay_mat")
        
        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                item.hinh_anh = res.get("secure_url", "")
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
        item.noi_nhat = request.form.get("noi_nhat")
        item.ngay_nhat = request.form.get("ngay_nhat")
        
        file = request.files.get("hinh_anh")
        if file and file.filename != "":
            try:
                res = cloudinary.uploader.upload(file)
                item.hinh_anh = res.get("secure_url", "")
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
        db_data = db.session.query(LostItem.noi_mat, func.count(LostItem.id)).group_by(LostItem.noi_mat).all()
        count_dict = dict(db_data)
    except Exception:
        count_dict = {}

    data = [(area, count_dict.get(area, 0)) for area in areas]
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
def match_items():
    page = max(1, request.args.get("page", 1, type=int))
    cat_filter = request.args.get("category", "").strip()
    per_page = 10
    try:
        lost_query = LostItem.query.filter(LostItem.trang_thai != "DA_TIM")
        found_query = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA")
        if cat_filter:
            lost_query = lost_query.filter(LostItem.loai_do == cat_filter)
            found_query = found_query.filter(FoundItem.loai_do == cat_filter)
        lost_items = lost_query.order_by(LostItem.id.desc()).all()
        found_items = found_query.order_by(FoundItem.id.desc()).all()

        # Đặc trưng được đọc từ DB; bản ghi cũ sẽ được xử lý bổ sung khi cần.
        lost_features = {x.id: get_item_features(x) for x in lost_items if check_real_image(x.hinh_anh)}
        found_features = {x.id: get_item_features(x) for x in found_items if check_real_image(x.hinh_anh)}

        ranked = []
        for lost in lost_items:
            for found in found_items:
                try:
                    if lost.loai_do and found.loai_do and lost.loai_do.strip().lower() != found.loai_do.strip().lower():
                        continue
                    key1, key2 = get_core_keyword(lost.ten_do), get_core_keyword(found.ten_do)
                    ten_sim = similarity(lost.ten_do, found.ten_do)
                    same_key = key1 == key2 and bool(key1)
                    name_score = 15 if same_key else int(ten_sim * 15)
                    c1, c2 = str(lost.mau_sac or "").strip(), str(found.mau_sac or "").strip()
                    color_score = int(similarity(c1, c2) * 15) if c1 and c2 else (10 if same_key else 0)
                    loc_score = 0
                    l1, l2 = str(lost.noi_mat or "").strip(), str(found.noi_nhat or "").strip()
                    if l1 and l2:
                        loc_sim = similarity(l1, l2)
                        loc_score = 10 if loc_sim > .8 else (5 if loc_sim > .5 else 0)
                    d_score = date_score(lost.ngay_mat, found.ngay_nhat)
                    base = name_score + color_score + loc_score + d_score

                    # Chỉ so ảnh khi có tín hiệu mô tả tối thiểu; nhưng chấm mọi cặp
                    # đủ điều kiện trước khi phân trang để xếp hạng toàn cục đúng.
                    img_score = 0
                    if (lost.id in lost_features and found.id in found_features
                            and (same_key or ten_sim >= 0.35)):
                        img_score = image_score_from_features(
                            lost_features[lost.id], found_features[found.id]
                        )[0]
                    total = base + img_score
                    label = ("🟢 Khả năng trùng khớp cao" if total >= 75 else
                             "🟡 Có khả năng trùng khớp" if total >= 55 else
                             "🟠 Có một số điểm tương đồng" if total >= 38 else
                             "🔴 Khả năng trùng khớp thấp")
                    ranked.append({
                        "lost": lost, "found": found, "score": total,
                        "ket_luan": label, "score_image": img_score,
                        "score_name": name_score, "score_color": color_score,
                        "score_location": loc_score, "score_date": d_score
                    })
                except Exception:
                    app.logger.exception("Lỗi tính cặp đối chiếu")
                    continue

        ranked.sort(key=lambda x: x["score"], reverse=True)
        total_matches = len(ranked)
        total_pages = max(1, (total_matches + per_page - 1) // per_page)
        page = min(page, total_pages)
        final_matches = ranked[(page - 1) * per_page:page * per_page]
        return render_template("match.html", matches=final_matches, page=page,
                               total_pages=total_pages, total_matches=total_matches,
                               selected_category=cat_filter)
    except Exception:
        db.session.rollback()
        app.logger.exception("Lỗi trang đối chiếu")
        return render_template("match.html", matches=[], page=1, total_pages=1,
                               total_matches=0, selected_category=cat_filter)

@app.route("/teacher-check/<int:lost_id>/<int:found_id>")
def teacher_check(lost_id, found_id):
    try:
        lost = LostItem.query.get_or_404(lost_id)
        found = FoundItem.query.get_or_404(found_id)
    except Exception:
        return redirect(url_for("match_items"))

    has1 = check_real_image(lost.hinh_anh)
    has2 = check_real_image(found.hinh_anh)

    key1 = get_core_keyword(lost.ten_do)
    key2 = get_core_keyword(found.ten_do)
    is_same_key = (key1 == key2 and key1 != "")

    if has1 and has2:
        anh_score, image_ratio = image_score_from_features(
            get_item_features(lost), get_item_features(found)
        )
    else:
        anh_score, image_ratio = 0, 0.0

    ten_score = 15 if is_same_key else int(similarity(lost.ten_do, found.ten_do) * 15)
    
    mau_score = 0
    m1 = str(lost.mau_sac or "").strip().lower()
    m2 = str(found.mau_sac or "").strip().lower()
    if m1 not in ["none", "null", ""] and m2 not in ["none", "null", ""]:
        mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)
    elif is_same_key:
        mau_score = 10

    vi_tri_score = 0
    v1 = str(lost.noi_mat or "").strip().lower()
    v2 = str(found.noi_nhat or "").strip().lower()
    if v1 not in ["none", "null", ""] and v2 not in ["none", "null", ""]:
        vi_tri = similarity(lost.noi_mat, found.noi_nhat)
        if vi_tri > 0.8:
            vi_tri_score = 10
        elif vi_tri > 0.5:
            vi_tri_score = 5

    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

    if not (has1 and has2):
        anh_score = 0

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
    if has1 and has2 and image_ratio < 0.20:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."
    elif not (has1 and has2):
        image_warning = "⚠️ Bài đăng này thiếu hình ảnh minh họa, vui lòng xác minh bằng thông tin mô tả."

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
            alert("Ma PIN Giao vien khong chinh xac! Vui long thu lai.");
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
            "Nơi mất": item.noi_mat,
            "Ngày mất": item.ngay_mat,
            "Trạng thái": "Đã tìm thấy" if item.trang_thai in ["DA_TIM", "Đã tìm thấy"] else "Đang tìm"
        } for i, item in enumerate(lost_items)]

        found_data = [{
            "STT": i + 1,
            "Tên đồ vật": item.ten_do,
            "Loại đồ": item.loai_do,
            "Màu sắc": item.mau_sac,
            "Nơi nhặt": item.noi_nhat,
            "Ngày nhặt": item.ngay_nhat,
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
