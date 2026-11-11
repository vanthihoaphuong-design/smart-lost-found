import os
import io
import uuid
import json
import re
import gc
import urllib3
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader

from PIL import Image, ImageOps, ImageFile

# Vô hiệu hóa cảnh báo SSL và cho phép PIL đọc ảnh truncated từ Cloudinary
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

VN_TZ = timezone(timedelta(hours=7))

# Bọc kiểm tra OpenCV an toàn chống sập Worker RAM
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

FEATURE_CACHE = {}
MAX_CACHE_SIZE = 100

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

def get_feature_cache(img_path):
    return FEATURE_CACHE.get(img_path) if img_path else None

def save_feature_cache(img_path, features):
    if img_path:
        if len(FEATURE_CACHE) > MAX_CACHE_SIZE:
            FEATURE_CACHE.clear()
        FEATURE_CACHE[img_path] = features

def calculate_image_features(img_path):
    """Tính toán Feature trích xuất từ OpenCV/PIL có hỗ trợ Cache & tối ưu RAM"""
    if not check_real_image(img_path):
        return None
    
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        data = None
        path_str = str(img_path).strip()
        
        # 1. Tải dữ liệu byte an toàn từ Cloudinary hoặc Local
        if path_str.startswith(("http://", "https://")):
            import requests
            resp = requests.get(path_str, timeout=3, verify=False)
            if resp.status_code == 200:
                data = resp.content
        else:
            p = os.path.join(app.config["UPLOAD_FOLDER"], path_str) if not os.path.isabs(path_str) else path_str
            if os.path.exists(p):
                with open(p, "rb") as f:
                    data = f.read()

        if not data:
            return None

        # 2. Ưu tiên xử lý OpenCV nếu thư viện khả dụng
        if HAS_OPENCV:
            try:
                arr = np.asarray(bytearray(data), dtype=np.uint8)
                img = cv2.imdecode(arr, -1)
                if img is not None:
                    # Tự động crop viền trắng nếu là ảnh sản phẩm/studio
                    gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    h_img, w_img = gray_temp.shape
                    corner_pixels = [gray_temp[0,0], gray_temp[0, w_img-1], gray_temp[h_img-1, 0], gray_temp[h_img-1, w_img-1]]
                    if sum(c > 235 for c in corner_pixels) >= 3:
                        _, thresh = cv2.threshold(gray_temp, 240, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            if w > 20 and h > 20:
                                img = img[y:y+h, x:x+w]

                    crop_img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    
                    orb = cv2.ORB_create(nfeatures=300)
                    kp, des = orb.detectAndCompute(gray, None)

                    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256])
                    if hist is not None:
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

                    features = {
                        "type": "opencv",
                        "orb_keypoints": kp,
                        "orb_descriptors": des,
                        "hsv_hist": hist
                    }
                    save_feature_cache(img_path, features)
                    return features
            except Exception as cv_err:
                print(f"Bỏ qua lỗi OpenCV, chuyển fallback PIL: {cv_err}")

        # 3. Phương án Fallback bằng PIL siêu nhẹ RAM
        with Image.open(io.BytesIO(data)) as im:
            im_rgb = im.convert("RGB")
            im_gray = im_rgb.convert("L")
            
            im_dh = im_gray.resize((17, 16), Image.Resampling.LANCZOS)
            pixels = list(im_dh.getdata())
            dhash_val = 0
            for r in range(16):
                for c in range(16):
                    left = pixels[r * 17 + c]
                    right = pixels[r * 17 + c + 1]
                    dhash_val = (dhash_val << 1) | int(left > right)

            im_thumb = im_rgb.resize((32, 32))
            hist = im_thumb.histogram()

            features = {
                "type": "pil",
                "dhash": dhash_val,
                "hist": hist
            }
            save_feature_cache(img_path, features)
            return features

    except Exception as e:
        print(f"Lỗi calculate_image_features ({img_path}): {e}")
        return None

def image_score(img1_filename, img2_filename, is_same_keyword=False):
    """Tính điểm tương đồng ảnh chính xác - Tối đa 50 điểm"""
    try:
        str1 = str(img1_filename).strip()
        str2 = str(img2_filename).strip()

        if str1 == str2:
            return 50, 1.0

        f1 = calculate_image_features(img1_filename)
        f2 = calculate_image_features(img2_filename)

        if f1 is None or f2 is None:
            fallback = 35 if is_same_keyword else 15
            return fallback, float(fallback / 50.0)

        # Xử lý nếu cả 2 có Feature dạng OpenCV
        if f1.get("type") == "opencv" and f2.get("type") == "opencv":
            good_count = 0
            try:
                kp1, des1 = f1.get("orb_keypoints", []), f1.get("orb_descriptors")
                kp2, des2 = f2.get("orb_keypoints", []), f2.get("orb_descriptors")
                if des1 is not None and des2 is not None and len(kp1) >= 2 and len(kp2) >= 2:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                    matches = bf.knnMatch(des1, des2, k=2)
                    for m_pair in matches:
                        if len(m_pair) == 2 and m_pair[0].distance < 0.78 * m_pair[1].distance:
                            good_count += 1
            except Exception:
                good_count = 0

            color_corr = 0.0
            try:
                hist1, hist2 = f1.get("hsv_hist"), f2.get("hsv_hist")
                if hist1 is not None and hist2 is not None:
                    color_corr = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))
            except Exception:
                color_corr = 0.0

            if good_count >= 10:
                score = 45 if is_same_keyword else 42
            elif good_count >= 5:
                score = 42 if is_same_keyword else 38
            elif good_count >= 3:
                score = 38 if is_same_keyword else 32
            elif good_count >= 1:
                score = 32 if is_same_keyword else 25
            else:
                score = int(color_corr * 25)

            score = min(50, max(0, score))
            return score, float(score / 50.0)

        # Xử lý điểm dHash + Histogram theo dạng PIL Fallback
        dist = (f1.get("dhash", 0) ^ f2.get("dhash", 0)).bit_count()
        s_shape = 25 if dist <= 60 else (20 if dist <= 90 else (15 if dist <= 120 else 8))

        h1, h2 = f1.get("hist", []), f2.get("hist", [])
        dot = sum(a * b for a, b in zip(h1, h2))
        norm1 = sum(a * a for a in h1) ** 0.5
        norm2 = sum(b * b for b in h2) ** 0.5
        c_sim = (dot / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
        s_color = int(c_sim * 25)

        total = s_shape + s_color
        if is_same_keyword and total < 30:
            total = 35

        total = min(50, max(0, total))
        return total, float(total / 50.0)

    except Exception as e:
        print(f"Lỗi image_score: {e}")
        fallback = 35 if is_same_keyword else 15
        return fallback, float(fallback / 50.0)

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
            except Exception as e:
                print(f"Lỗi upload Cloudinary: {e}")
            
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
            except Exception as e:
                print(f"Lỗi upload Cloudinary: {e}")
            
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
    matches = []
    try:
        # Lấy tối đa 25 bài chưa hoàn trả mới nhất
        lost_items = LostItem.query.filter(LostItem.trang_thai != "DA_TIM").order_by(LostItem.id.desc()).limit(25).all()
        found_items = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA").order_by(FoundItem.id.desc()).limit(25).all()
    except Exception:
        lost_items, found_items = [], []

    for lost in lost_items:
        for found in found_items:
            try:
                key1 = get_core_keyword(lost.ten_do)
                key2 = get_core_keyword(found.ten_do)
                ten_sim = similarity(lost.ten_do, found.ten_do)
                is_same_key = (key1 == key2 and key1 != "")

                has1 = check_real_image(lost.hinh_anh)
                has2 = check_real_image(found.hinh_anh)

                # 1. ĐIỂM HÌNH ẢNH (Tối đa 50đ)
                if has1 and has2:
                    anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, is_same_keyword=is_same_key)
                else:
                    anh_score, image_ratio = 0, 0.0

                # 2. ĐIỂM TÊN ĐỒ (Tối đa 15đ)
                ten_score = 15 if is_same_key else int(ten_sim * 15)

                # 3. ĐIỂM MÀU SẮC (Tối đa 15đ)
                mau_score = 0
                m1 = str(lost.mau_sac or "").strip().lower()
                m2 = str(found.mau_sac or "").strip().lower()
                if m1 not in ["none", "null", ""] and m2 not in ["none", "null", ""]:
                    mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)
                elif is_same_key:
                    mau_score = 12

                # 4. ĐIỂM VỊ TRÍ (Tối đa 10đ)
                vi_tri_score = 0
                v1 = str(lost.noi_mat or "").strip().lower()
                v2 = str(found.noi_nhat or "").strip().lower()
                if v1 not in ["none", "null", ""] and v2 not in ["none", "null", ""]:
                    vi_tri = similarity(lost.noi_mat, found.noi_nhat)
                    if vi_tri > 0.8:
                        vi_tri_score = 10
                    elif vi_tri > 0.5:
                        vi_tri_score = 5

                # 5. ĐIỂM NGÀY THÁNG (Tối đa 10đ)
                ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

                if not (has1 and has2):
                    anh_score = 0

                # TỔNG ĐIỂM
                score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score
                
                # BỘ LỌC HIỂN THỊ: Chỉ hiển thị các cặp có điểm tổng từ 25/100 trở lên
                if score < 25:
                    continue

                image_confidence = int(min(100, image_ratio * 100))

                reasons = []
                if has1 and has2 and anh_score >= 20:
                    reasons.append("Hình ảnh có điểm tương đồng")
                if ten_score >= 10:
                    reasons.append("Tên đồ tương đồng")
                if mau_score >= 10:
                    reasons.append("Màu sắc tương đồng")
                if vi_tri_score >= 5:
                    reasons.append("Vị trí liên quan")
                if ngay_score >= 6:
                    reasons.append("Thời gian khớp/gần nhau")

                if score >= 75:
                    ket_luan = "🟢 Khả năng trùng khớp cao"
                elif score >= 55:
                    ket_luan = "🟡 Có khả năng trùng khớp"
                elif score >= 38:
                    ket_luan = "🟠 Có một số điểm tương đồng"
                else:
                    ket_luan = "🔴 Khả năng trùng khớp thấp"

                details = {
                    "hinh_anh": anh_score,
                    "ten_do": ten_score,
                    "mau_sac": mau_score,
                    "vi_tri": vi_tri_score,
                    "ngay": ngay_score,
                    "score_image": anh_score,
                    "score_name": ten_score,
                    "score_color": mau_score,
                    "score_location": vi_tri_score,
                    "score_date": ngay_score,
                    "image_confidence": image_confidence
                }

                matches.append({
                    "lost": lost,
                    "found": found,
                    "score": score,
                    "ket_luan": ket_luan,
                    "reason": reasons,
                    "details": details,
                    "score_image": anh_score,
                    "score_name": ten_score,
                    "score_color": mau_score,
                    "score_location": vi_tri_score,
                    "score_date": ngay_score,
                    "hinh_anh": anh_score,
                    "ten_do": ten_score,
                    "mau_sac": mau_score,
                    "vi_tri": vi_tri_score,
                    "ngay": ngay_score
                })
            except Exception:
                continue

    matches.sort(key=lambda x: x["score"], reverse=True)
    gc.collect()
    return render_template("match.html", matches=matches)
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
        anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, is_same_keyword=is_same_key)
    else:
        anh_score, image_ratio = 0, 0.0

    ten_score = 15 if is_same_key else int(similarity(lost.ten_do, found.ten_do) * 15)
    
    mau_score = 0
    m1 = str(lost.mau_sac or "").strip().lower()
    m2 = str(found.mau_sac or "").strip().lower()
    if m1 not in ["none", "null", ""] and m2 not in ["none", "null", ""]:
        mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)
    elif is_same_key:
        mau_score = 12

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
        import pandas as pd
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
    except Exception as e:
        print("Lỗi xuất Excel:", e)
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
