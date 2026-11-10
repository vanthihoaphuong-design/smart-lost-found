import os
import io
import uuid
import json
import re
import gc
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader

VN_TZ = timezone(timedelta(hours=7))

cloudinary.config(
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME", ""),
    api_key = os.environ.get("CLOUDINARY_API_KEY", ""),
    api_secret = os.environ.get("CLOUDINARY_API_SECRET", ""),
    secure = True
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smart_lost_found_secret_key_123")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db_url = os.environ.get("DATABASE_URL", "sqlite:///lost_found.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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

STOP_WORDS_VN = {"con", "cái", "chiếc", "hộp", "bộ", "cặp", "đôi", "quả", "trái", "tấm", "bức", "tờ", "cục"}

def check_real_image(img_path):
    if not img_path:
        return False
    try:
        str_path = str(img_path).strip().lower()
        if str_path in ["", "none", "null", "undefined"]:
            return False
        invalid_keywords = ["chua co anh", "chưa có ảnh", "no_image", "default", "placeholder", "120?text="]
        if any(k in str_path for k in invalid_keywords):
            return False
        if str_path.startswith("http://") or str_path.startswith("https://"):
            return True
        p = os.path.join(app.config.get("UPLOAD_FOLDER", "static/uploads"), str_path) if not os.path.isabs(str_path) else str_path
        return os.path.exists(p)
    except Exception:
        return False

def _read_image_bytes(img_path, max_bytes=1_500_000):
    """Đọc ảnh có giới hạn dung lượng 1.5MB để tối ưu 512MB RAM Render."""
    if not check_real_image(img_path):
        return None

    try:
        path = str(img_path).strip()
        if path.startswith(("http://", "https://")):
            import urllib.request
            req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                return None
            return data

        local_path = path if os.path.isabs(path) else os.path.join(
            app.config.get("UPLOAD_FOLDER", "static/uploads"), path
        )
        if not os.path.isfile(local_path):
            return None
        if os.path.getsize(local_path) > max_bytes:
            return None
        with open(local_path, "rb") as f:
            return f.read(max_bytes + 1)
    except Exception as e:
        print(f"Không đọc được ảnh: {e}")
        return None

def _get_image_fingerprint(img_path):
    """
    Trích xuất đặc trưng hình học, cấu trúc và dải màu với Square Pad bảo toàn tỉ lệ.
    """
    data = _read_image_bytes(img_path)
    if not data:
        return None

    try:
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(data)) as im:
            im_rgb = im.convert("RGB")
            
            # Đệm viền trắng vuông cho khung hình để chuẩn hóa tỷ lệ vật thể
            w, h = im_rgb.size
            max_side = max(w, h)
            pad_w = (max_side - w) // 2
            pad_h = (max_side - h) // 2
            im_square = ImageOps.expand(im_rgb, (pad_w, pad_h, pad_w, pad_h), fill=(255, 255, 255))
            sw, sh = im_square.size

            # 1. dHash Khung thể toàn cảnh (16x16 -> 256 bits)
            im_gray = im_square.convert("L")
            im_dh = im_gray.resize((17, 16), Image.Resampling.LANCZOS)
            pixels_dh = list(im_dh.getdata())
            dhash_val = 0
            for row in range(16):
                for col in range(16):
                    left = pixels_dh[row * 17 + col]
                    right = pixels_dh[row * 17 + col + 1]
                    dhash_val = (dhash_val << 1) | int(left > right)

            # 2. dHash VÙNG TRUNG TÂM (Cắt 50% ruột trung tâm để giảm nhiễu phông nền)
            crop_center = im_gray.crop((int(sw * 0.25), int(sh * 0.25), int(sw * 0.75), int(sh * 0.75)))
            im_dh_c = crop_center.resize((17, 16), Image.Resampling.LANCZOS)
            pixels_dh_c = list(im_dh_c.getdata())
            dhash_center = 0
            for row in range(16):
                for col in range(16):
                    left = pixels_dh_c[row * 17 + col]
                    right = pixels_dh_c[row * 17 + col + 1]
                    dhash_center = (dhash_center << 1) | int(left > right)

            # 3. Color Spectrum Histogram (32x32 = 1024 dải màu nhạy cao)
            im_thumb = im_square.resize((32, 32))
            hist_color = im_thumb.histogram()

            return {
                "dhash": dhash_val,
                "dhash_center": dhash_center,
                "hist": hist_color
            }
    except Exception as e:
        print(f"Lỗi tính fingerprint: {e}")
        return None
    finally:
        del data

def image_score(img1_filename, img2_filename, is_same_keyword=False):
    """
    HÀM ĐỐI SÁNH HÌNH ÁNH ĐỘC LẬP (Mịn thang điểm, không bị phạt rớt 0đ do ánh sáng phông nền):
    - Khung dáng toàn cảnh: Max 20đ
    - Cấu trúc tâm vật thể: Max 15đ
    - Tương đồng dải màu: Max 15đ
    * KHÔNG cộng điểm thưởng từ tên gọi đồ vật!
    """
    try:
        if not check_real_image(img1_filename) or not check_real_image(img2_filename):
            return 0, 0.0

        if str(img1_filename).strip() == str(img2_filename).strip():
            return 50, 1.0

        f1 = _get_image_fingerprint(img1_filename)
        f2 = _get_image_fingerprint(img2_filename)

        if f1 is None or f2 is None:
            return 0, 0.0

        # 1. ĐÁNH GIÁ KHUNG DÁNG HÌNH HỌC (Max 20 điểm)
        dist_dh = (f1["dhash"] ^ f2["dhash"]).bit_count()
        if dist_dh <= 70:
            score_shape = 20
        elif dist_dh <= 100:
            score_shape = 16
        elif dist_dh <= 130:
            score_shape = 12
        elif dist_dh <= 160:
            score_shape = 8
        elif dist_dh <= 185:
            score_shape = 4
        else:
            score_shape = 1

        # 2. ĐÁNH GIÁ CẤU TRÚC RUỘT/TÂM VẬT THỂ (Max 15 điểm)
        dist_center = (f1["dhash_center"] ^ f2["dhash_center"]).bit_count()
        if dist_center <= 60:
            score_structure = 15
        elif dist_center <= 90:
            score_structure = 12
        elif dist_center <= 120:
            score_structure = 9
        elif dist_center <= 150:
            score_structure = 5
        else:
            score_structure = 2

        # 3. ĐÁNH GIÁ DẢI MÀU RGB (Max 15 điểm)
        h1, h2 = f1["hist"], f2["hist"]
        dot = sum(a * b for a, b in zip(h1, h2))
        norm1 = sum(a * a for a in h1) ** 0.5
        norm2 = sum(b * b for b in h2) ** 0.5
        color_sim = (dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0

        if color_sim >= 0.75:
            score_color = 15
        elif color_sim >= 0.60:
            score_color = 12
        elif color_sim >= 0.45:
            score_color = 8
        elif color_sim >= 0.30:
            score_color = 4
        else:
            score_color = 1

        # TỔNG HỢP ĐIỂM HÌNH ÁNH THUẦN THỊ GIÁC
        total_image_score = score_shape + score_structure + score_color
        total_image_score = min(50, max(0, total_image_score))
        similarity_ratio = float(total_image_score / 50.0)

        return total_image_score, similarity_ratio
    except Exception as e:
        print(f"Lỗi image_score: {e}")
        return 0, 0.0

def get_core_keyword(text_name):
    if not text_name:
        return ""
    t = str(text_name).lower().strip()
    synonyms_map = {
        "máy tính casio": ["casio", "fx580", "fx 580", "máy tính bỏ túi", "casio580"],
        "laptop": ["laptop", "macbook", "máy tính xách tay"],
        "chuột": ["chuột", "con chuột", "chuột máy tính", "mouse"],
        "sách toán": ["sách toán", "toán", "math", "bài tập toán"],
        "sách văn": ["sách văn", "ngữ văn", "văn học", "bài tập ngữ văn"],
        "sách tiếng anh": ["sách anh", "tiếng anh", "english"],
        "sách tin học": ["tin học", "tin 11", "tin 9", "tin 10"],
        "sách lý": ["vật lý", "lý"],
        "sách hóa": ["hóa học", "hóa"],
        "bút": ["bút", "viết", "bút bi", "bút chì", "bút xóa"],
        "điện thoại": ["điện thoại", "phone", "iphone", "samsung"],
        "balo": ["balo", "ba lô", "túi", "túi xách", "cặp"],
        "ví": ["ví", "bóp", "ví tiền", "bóp tiền"],
        "dây chuyền": ["dây chuyền", "vòng cổ"],
        "nhẫn": ["nhẫn"],
        "nón": ["nón", "mũ", "nón kết"],
        "thẻ": ["thẻ", "căn cước", "cccd", "thẻ học sinh"],
        "bình nước": ["bình nước", "bình giữ nhiệt"],
        "sữa": ["sữa", "hộp sữa", "milo", "vinamilk", "th true milk"],
        "sáo": ["sáo", "sáo trúc"],
        "đèn": ["đèn", "đèn học", "đèn để bàn"]
    }

    for core_key, words in synonyms_map.items():
        if any(re.search(r'\b' + re.escape(w) + r'\b', t) for w in words):
            return core_key
            
    tokens = [w for w in t.split() if w not in STOP_WORDS_VN]
    return tokens[0] if tokens else t

def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()
    if t1 == t2:
        return 1.0
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
    except Exception as e:
        print("Lỗi lưu PIN:", e)
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
            except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
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
