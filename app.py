import os
import io
import uuid
import json
import re
import gc
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
    except Exception as e:
        print(f"Lỗi Create DB All: {e}")

FEATURE_CACHE = {}
MAX_CACHE_SIZE = 512

STOP_WORDS_VN = {"con", "cái", "chiếc", "hộp", "bộ", "cặp", "đôi", "quả", "trái", "tấm", "bức", "tờ", "cục"}
COLOR_WORDS_VN = ["đen", "đỏ", "xanh", "vàng", "trắng", "hồng", "tím", "nâu", "xám", "cam", "bạc"]


def extract_colors(text):
    if not text:
        return set()
    t = str(text).lower()
    return {c for c in COLOR_WORDS_VN if re.search(rf'\b{c}\b', t)}


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
    if len(FEATURE_CACHE) >= MAX_CACHE_SIZE and key not in FEATURE_CACHE:
        FEATURE_CACHE.pop(next(iter(FEATURE_CACHE)))
    FEATURE_CACHE[key] = features


def _read_image_bytes(img_path, max_bytes=8 * 1024 * 1024):
    """Đọc toàn bộ ảnh trong giới hạn kích thước."""
    value = str(img_path).strip()
    if value.startswith(("http://", "https://")):
        import requests
        response = requests.get(
            value, timeout=(3, 8), stream=True, verify=False,
            headers={"User-Agent": "SmartLostFound/1.0"}
        )
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                response.close()
                return None
            chunks.append(chunk)
        response.close()
        return b"".join(chunks)

    path = value if os.path.isabs(value) else os.path.join(
        app.config.get("UPLOAD_FOLDER", "static/uploads"), value
    )
    if not os.path.isfile(path) or os.path.getsize(path) > max_bytes:
        return None
    with open(path, "rb") as file:
        return file.read(max_bytes + 1)


def _phash_from_gray(gray):
    """Perceptual hash 64-bit dựa trên DCT, ổn định hơn dHash ở ảnh resize."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")
    dct = cv2.dct(small)
    low = dct[:8, :8].flatten()
    values = low[1:]
    median = float(np.median(values))
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def calculate_image_features(img_path):
    """Trích xuất đặc trưng trên ảnh thu nhỏ 320x320 để tối ưu RAM và tốc độ."""
    if not check_real_image(img_path):
        return None

    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        data = _read_image_bytes(img_path, max_bytes=5 * 1024 * 1024)
        if not data:
            return None

        with Image.open(io.BytesIO(data)) as source:
            try:
                source.draft("RGB", (320, 320))
            except Exception:
                pass
            width, height = source.size
            if width < 2 or height < 2 or width * height > 12_000_000:
                return None
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((320, 320), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (320, 320), (245, 245, 245))
            canvas.paste(image, ((320 - image.width) // 2, (320 - image.height) // 2))
            rgb = np.asarray(canvas, dtype=np.uint8) if HAS_OPENCV else None

        if HAS_OPENCV and rgb is not None:
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(
                nfeatures=250, scaleFactor=1.2, nlevels=6,
                edgeThreshold=15, patchSize=31, fastThreshold=14
            )
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            # Tính histogram màu tập trung vùng trung tâm (60% giữa ảnh) để giảm nhiễu nền gạch/bàn
            h, w = image_bgr.shape[:2]
            center_crop = image_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
            cv2.normalize(color_hist, color_hist, 1.0, 0.0, cv2.NORM_L1)

            features = {
                "version": 5,
                "type": "opencv_v5_thumbnail",
                "phash": _phash_from_gray(gray),
                "orb_descriptors": descriptors,
                "orb_count": len(keypoints) if keypoints else 0,
                "color_hist": color_hist,
            }
        else:
            tiny = canvas.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            px = list(tiny.getdata())
            dhash = 0
            for row in range(8):
                for col in range(8):
                    dhash = (dhash << 1) | int(px[row * 9 + col] > px[row * 9 + col + 1])
            hist = canvas.histogram()
            features = {"version": 2, "type": "pil_v2", "dhash": dhash, "hist": hist}

        save_feature_cache(img_path, features)
        return features
    except Exception as exc:
        print(f"Lỗi trích xuất đặc trưng ảnh: {exc}")
        return None


def image_score(img1_filename, img2_filename, is_same_keyword=False):
    """Chấm điểm hình ảnh (0-50) tối ưu cho ảnh chụp thực tế khác góc/khác nền."""
    try:
        if not check_real_image(img1_filename) or not check_real_image(img2_filename):
            return 0, 0.0

        if str(img1_filename).strip() == str(img2_filename).strip():
            return 50, 1.0

        f1 = calculate_image_features(img1_filename)
        f2 = calculate_image_features(img2_filename)
        if f1 is None or f2 is None:
            return 0, 0.0

        if f1.get("type", "").startswith("opencv_v") and f2.get("type", "").startswith("opencv_v"):
            # 1. pHash (tối đa 20đ): nhận diện bố cục tổng thể
            hamming = (int(f1["phash"]) ^ int(f2["phash"])).bit_count()
            phash_score = max(0.0, 1.0 - hamming / 63.0)
            if phash_score >= 0.82:
                shape_points = 20
            elif phash_score >= 0.70:
                shape_points = 16
            elif phash_score >= 0.60:
                shape_points = 12
            elif phash_score >= 0.50:
                shape_points = 7
            else:
                shape_points = 0

            # 2. ORB (tối đa 20đ): đối chiếu đặc trưng cục bộ 2 chiều
            d1, d2 = f1.get("orb_descriptors"), f2.get("orb_descriptors")
            orb_points = 0
            if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                forward = matcher.knnMatch(d1, d2, k=2)
                reverse = matcher.knnMatch(d2, d1, k=2)

                def _ratio_matches(pairs):
                    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]

                good_f = _ratio_matches(forward)
                good_r = _ratio_matches(reverse)
                reverse_pairs = {(m.trainIdx, m.queryIdx) for m in good_r}
                mutual = sum((m.queryIdx, m.trainIdx) in reverse_pairs for m in good_f)

                if mutual >= 12:
                    orb_points = 20
                elif mutual >= 7:
                    orb_points = 16
                elif mutual >= 4:
                    orb_points = 11
                elif mutual >= 2:
                    orb_points = 6

            # 3. Màu sắc vùng trung tâm HSV Histogram (tối đa 10đ)
            color_points = 0
            hist1, hist2 = f1.get("color_hist"), f2.get("color_hist")
            if hist1 is not None and hist2 is not None:
                corr = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
                if corr >= 0.78:
                    color_points = 10
                elif corr >= 0.58:
                    color_points = 7
                elif corr >= 0.40:
                    color_points = 4

            # Cộng thưởng nhẹ (+5đ) nếu tên đồ trùng từ khóa cốt lõi và ảnh có nét tương đồng
            bonus = 5 if (is_same_keyword and (shape_points + orb_points + color_points) >= 18) else 0

            total = min(50, shape_points + orb_points + color_points + bonus)
            return int(total), float(total / 50.0)

        # Fallback PIL
        if f1.get("type") == "pil_v2" and f2.get("type") == "pil_v2":
            dist = (int(f1.get("dhash", 0)) ^ int(f2.get("dhash", 0))).bit_count()
            shape = 35 if dist <= 12 else (25 if dist <= 20 else (15 if dist <= 28 else 7))
            h1, h2 = f1.get("hist", []), f2.get("hist", [])
            dot = sum(a * b for a, b in zip(h1, h2))
            n1 = sum(a * a for a in h1) ** 0.5
            n2 = sum(b * b for b in h2) ** 0.5
            sim_c = dot / (n1 * n2) if n1 and n2 else 0.0
            color = 15 if sim_c >= 0.80 else (10 if sim_c >= 0.65 else 5)
            total = min(50, shape + color)
            return int(total), float(total / 50.0)

        return 0, 0.0
    except Exception as exc:
        print(f"Lỗi tính điểm hình ảnh: {exc}")
        return 0, 0.0


def get_core_keyword(text_name):
    if not text_name:
        return ""
    t = str(text_name).lower().strip()

    synonyms_map = {
        "máy tính casio": ["casio", "fx580", "fx 580", "máy tính bỏ túi", "casio580"],
        "laptop": ["laptop", "macbook", "máy tính xách tay"],
        "chuột": ["chuột", "con chuột", "chuột máy tính", "mouse"],
        "sách toán": ["sách toán", "sbt toán", "toán"],
        "sách văn": ["sách văn", "ngữ văn", "văn học"],
        "sách tiếng anh": ["sách anh", "tiếng anh", "english"],
        "sách tin học": ["tin học", "sách tin", "tin 12", "tin 11", "tin 10", "tin 9", "tin 8", "tin 7", "tin 6"],
        "sách công nghệ": ["công nghệ", "sách công nghệ"],
        "sách lý": ["vật lý", "sách lý"],
        "sách hóa": ["hóa học", "sách hóa"],
        "bút bi": ["bút bi", "viết bi"],
        "bút chì": ["bút chì", "bút chì kim"],
        "bút": ["cây bút", "bút mực", "bút nước", "bút gel", "bút"],
        "thước": ["cây thước", "thước kẻ", "thước"],
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
            # Gắn kèm số lớp/mã số nếu có (VD: "sách tin học_12" khác hẳn "sách tin học_8")
            nums = re.findall(r'\d+', t)
            return f"{core_key}_{'_'.join(nums)}" if nums else core_key

    tokens = [w for w in t.split() if w not in STOP_WORDS_VN]
    return tokens[0] if tokens else t


def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()

    if t1 == t2:
        return 1.0

    # 1. KIỂM TRA SỐ HIỆU / KHỐI LỚP TRƯỚC TIÊN (Sách tin 12 vs Sách tin 8 -> loại ngay)
    nums1 = re.findall(r'\d+', t1)
    nums2 = re.findall(r'\d+', t2)
    if nums1 and nums2 and nums1 != nums2:
        return 0.05

    # 2. KIỂM TRA XUNG ĐỘT MÀU SẮC NGAY TRONG TÊN ĐỒ (Bút đen vs Cây bút đỏ -> phạt nặng)
    c1 = extract_colors(t1)
    c2 = extract_colors(t2)
    if c1 and c2 and not c1.intersection(c2):
        return 0.15

    k1 = get_core_keyword(t1)
    k2 = get_core_keyword(t2)

    if k1 == k2 and k1 != "":
        return 0.92

    if t1 in t2 or t2 in t1:
        return 0.85

    return SequenceMatcher(None, t1, t2).ratio()


def evaluate_color_match(lost_item, found_item, is_same_key):
    """Tính điểm màu sắc (0-15) và kiểm tra xem 2 đồ vật có trái màu nhau không."""
    m1 = str(lost_item.mau_sac or "").strip().lower()
    m2 = str(found_item.mau_sac or "").strip().lower()
    has_m1 = m1 not in ["none", "null", ""]
    has_m2 = m2 not in ["none", "null", ""]

    # Gộp màu từ cả ô màu sắc và tên đồ (đề phòng người dùng ghi màu vào tên như "Bút đen")
    colors1 = extract_colors(m1) | extract_colors(lost_item.ten_do)
    colors2 = extract_colors(m2) | extract_colors(found_item.ten_do)

    if colors1 and colors2:
        if colors1.intersection(colors2):
            return 15, False  # Trùng màu, không xung đột
        else:
            return 0, True    # Trái màu hoàn toàn (VD: đen vs đỏ)

    if has_m1 and has_m2:
        sim = similarity(m1, m2)
        return int(sim * 15), (sim < 0.3)

    return (10 if is_same_key else 0), False


def location_score(loc1, loc2):
    """Chấm điểm vị trí (0-10): nhận diện chính xác trùng phòng và các phòng/lớp liền kề."""
    v1 = str(loc1 or "").strip().lower()
    v2 = str(loc2 or "").strip().lower()
    if v1 in ["none", "null", ""] or v2 in ["none", "null", ""]:
        return 0
    if v1 == v2:
        return 10

    # Bỏ số phòng/lớp để kiểm tra cùng khu/dãy (VD: "Phòng Tin học 1" và "Phòng Tin học 2")
    base1 = re.sub(r'\d+', '', v1).strip()
    base2 = re.sub(r'\d+', '', v2).strip()
    if base1 and base1 == base2:
        return 7

    # Cùng khối lớp (VD: 8A1 và 8A2)
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
@app.route("/match/<int:item_id>")
def match_items(item_id=None):
    """Đối sánh 2 tầng: Lọc nhanh bằng chữ & loại xung đột số lớp/màu sắc trước, sau đó phân tích AI ảnh."""
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
        for obj in found_items:
            obj.has_image = check_real_image(obj.hinh_anh)

        # Tầng 1: Chấm điểm văn bản trong RAM
        candidates = []
        sequence = 0

        for lost in lost_items:
            key1 = get_core_keyword(lost.ten_do)
            for found in found_items:
                try:
                    if (lost.loai_do and found.loai_do and
                            str(lost.loai_do).strip().lower() != str(found.loai_do).strip().lower()):
                        continue

                    key2 = get_core_keyword(found.ten_do)
                    ten_sim = similarity(lost.ten_do, found.ten_do)
                    is_same_key = (key1 == key2 and key1 != "")

                    # Loại bỏ ngay nếu khác số lớp/khối (VD: Tin 12 vs Tin 8) hoặc quá khác tên
                    if not (is_same_key or ten_sim >= 0.35):
                        continue

                    ten_score = 15 if is_same_key else int(ten_sim * 15)
                    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key)

                    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
                    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
                    base_score = ten_score + mau_score + vi_tri_score + ngay_score

                    sequence += 1
                    candidates.append({
                        "lost": lost,
                        "found": found,
                        "base_score": base_score,
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

        candidates.sort(key=lambda x: (x["base_score"], x["seq"]), reverse=True)
        total_matches = len(candidates)
        total_pages = max(1, (total_matches + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        # Chỉ phân tích AI ảnh cho top (page * 10 + 15) cặp tiềm năng nhất
        eval_limit = min(total_matches, page * per_page + 15)
        evaluated_matches = []

        for item in candidates[:eval_limit]:
            lost, found = item["lost"], item["found"]
            if lost.has_image and found.has_image:
                anh_score, _ = image_score(
                    lost.hinh_anh, found.hinh_anh,
                    is_same_keyword=item["is_same_key"]
                )
                # Nếu 2 món trái màu hoàn toàn (VD: Bút đen vs Bút đỏ), giảm điểm ảnh để tránh cộng lây nền gạch
                if item["color_conflict"]:
                    anh_score = int(anh_score * 0.35)
            else:
                anh_score = 0

            total_score = item["base_score"] + anh_score
            if total_score >= 75:
                ket_luan = "🟢 Khả năng trùng khớp cao"
            elif total_score >= 55:
                ket_luan = "🟡 Có khả năng trùng khớp"
            elif total_score >= 38:
                ket_luan = "🟠 Có một số điểm tương đồng"
            else:
                ket_luan = "🔴 Khả năng trùng khớp thấp"

            evaluated_matches.append({
                "lost": lost,
                "found": found,
                "score": total_score,
                "ket_luan": ket_luan,
                "score_image": anh_score,
                "score_name": item["score_name"],
                "score_color": item["score_color"],
                "score_location": item["score_location"],
                "score_date": item["score_date"],
            })

        evaluated_matches.sort(key=lambda x: x["score"], reverse=True)
        start_idx = (page - 1) * per_page
        page_matches = evaluated_matches[start_idx:start_idx + per_page]

        FEATURE_CACHE.clear()
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
        FEATURE_CACHE.clear()
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
    except Exception:
        return redirect(url_for("match_items"))

    has1 = check_real_image(lost.hinh_anh)
    has2 = check_real_image(found.hinh_anh)

    key1 = get_core_keyword(lost.ten_do)
    key2 = get_core_keyword(found.ten_do)
    ten_sim = similarity(lost.ten_do, found.ten_do)
    is_same_key = (key1 == key2 and key1 != "")

    ten_score = 15 if is_same_key else int(ten_sim * 15)
    mau_score, color_conflict = evaluate_color_match(lost, found, is_same_key)

    if has1 and has2:
        anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, is_same_keyword=is_same_key)
        if color_conflict:
            anh_score = int(anh_score * 0.35)
            image_ratio = anh_score / 50.0
    else:
        anh_score, image_ratio = 0, 0.0

    vi_tri_score = location_score(lost.noi_mat, found.noi_nhat)
    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

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
        image_warning = "⚠️ Hai món đồ có màu sắc khác nhau, giáo viên cần đối chiếu kỹ."
    elif has1 and has2 and image_ratio < 0.25:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."
    elif not (has1 and has2):
        image_warning = "⚠️ Bài đăng này thiếu hình ảnh minh họa, vui lòng xác minh bằng thông tin mô tả."

    FEATURE_CACHE.clear()
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
