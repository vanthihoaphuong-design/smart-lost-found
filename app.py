import os
import io
import uuid
import json
from datetime import datetime, timezone, timedelta
from PIL import Image
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader

VN_TZ = timezone(timedelta(hours=7))

try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except Exception as e:
    HAS_OPENCV = False
    print(f"Lỗi OpenCV: {e}")

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

FEATURE_CACHE = {}

def check_real_image(img_path):
    if not img_path:
        return False
    try:
        str_path = str(img_path).strip().lower()
        if str_path in ["", "none", "null", "undefined"]:
            return False
            
        invalid_keywords = [
            "chua co anh", "chưa có ảnh", "no_image", "default", 
            "placeholder", "120?text="
        ]
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
        FEATURE_CACHE[img_path] = features

def calculate_image_features(img_path):
    if not HAS_OPENCV or not check_real_image(img_path):
        return None
    
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        img = None
        if str(img_path).startswith("http://") or str(img_path).startswith("https://"):
            import urllib.request
            req = urllib.request.urlopen(str(img_path), timeout=5)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            p = os.path.join(app.config["UPLOAD_FOLDER"], str(img_path)) if not os.path.isabs(str(img_path)) else str(img_path)
            if os.path.exists(p):
                img = cv2.imread(p)

        if img is None:
            return None

        # Resize ảnh về kích thước chuẩn
        crop_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        features = {}

        # 1. ORB Keypoints & Descriptors
        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp, des = orb.detectAndCompute(gray, None)
            features["orb_keypoints"] = kp
            features["orb_descriptors"] = des
        except Exception:
            features["orb_keypoints"], features["orb_descriptors"] = [], None

        # 2. HSV Color Histogram (Tránh nhiễu màu phông trắng)
        try:
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 20], [0, 180, 0, 256])
            if hist is not None:
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            features["hsv_hist"] = hist
        except Exception:
            features["hsv_hist"] = None

        save_feature_cache(img_path, features)
        return features
    except Exception as e:
        print(f"Lỗi tính feature: {e}")
        return None

def image_score(img1_filename, img2_filename):
    try:
        if not check_real_image(img1_filename) or not check_real_image(img2_filename):
            return 0, 0.0

        str1 = str(img1_filename).strip()
        str2 = str(img2_filename).strip()

        if str1 == str2:
            return 50, 1.0

        features1 = calculate_image_features(img1_filename)
        features2 = calculate_image_features(img2_filename)

        if features1 is None or features2 is None:
            return 0, 0.0

        good_count = 0
        try:
            kp1, des1 = features1.get("orb_keypoints", []), features1.get("orb_descriptors")
            kp2, des2 = features2.get("orb_keypoints", []), features2.get("orb_descriptors")
            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)
                for m_pair in matches:
                    if len(m_pair) == 2 and m_pair[0].distance < 0.70 * m_pair[1].distance:
                        good_count += 1
        except Exception:
            good_count = 0

        color_corr = 0.0
        try:
            hist1, hist2 = features1.get("hsv_hist"), features2.get("hsv_hist")
            if hist1 is not None and hist2 is not None:
                color_corr = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))
        except Exception:
            color_corr = 0.0

        # Điểm ảnh tính chính xác dựa trên điểm đặc trưng vật thể thực sự
        if good_count >= 15:
            score = 50
        elif good_count >= 10:
            score = 40
        elif good_count >= 5:
            score = 30
        elif good_count >= 2:
            score = 20
        else:
            score = int(color_corr * 15)

        score = min(50, max(0, score))
        return score, float(score / 50.0)
    except Exception as e:
        print(f"Lỗi image_score: {e}")
        return 0, 0.0

def get_base_category(text_name, category_str):
    """LỚP 1: Tự động gom nhóm danh mục thực sự của đồ vật"""
    t = (str(text_name) + " " + str(category_str)).lower().strip()

    if any(k in t for k in ["laptop", "máy tính", "điện thoại", "phone", "casio", "sạc", "airpod", "ipad"]):
        return "ELECTRONICS"
    if any(k in t for k in ["bút", "viết", "sách", "vở", "tập", "thước", "hộp bút", "bóp viết"]):
        return "STATIONERY"
    if any(k in t for k in ["balo", "ba lô", "ví", "bóp", "túi", "áo", "quần", "dây chuyền", "nhẫn", "vòng", "nón", "mũ"]):
        return "PERSONAL_ITEMS"
    if any(k in t for k in ["thẻ", "căn cước", "cccd", "giấy"]):
        return "DOCUMENTS"
    
    return "OTHER"

def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()
    if t1 == t2:
        return 1.0

    synonyms = [
        {"ví", "bóp", "ví tiền", "bóp tiền"},
        {"cặp", "balo", "ba lô", "túi"},
        {"bút", "viết", "bút bi"},
        {"thẻ", "thẻ học sinh", "thẻ hs"},
        {"máy tính", "casio", "máy tính bỏ túi"},
        {"nón", "mũ", "nón kết"}
    ]
    
    for group in synonyms:
        if any(w in t1 for w in group) and any(w in t2 for w in group):
            return 0.90

    stopwords = ["sách", "bài tập", "bút", "cái", "chiếc", "vở", "tập"]
    w1 = [w for w in t1.split() if w not in stopwords]
    w2 = [w for w in t2.split() if w not in stopwords]
    
    if t1 in t2 or t2 in t1:
        return 0.85
        
    if w1 and w2 and set(w1).isdisjoint(set(w2)):
        return 0.1

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
    return 10 if diff == 0 else (8 if diff == 1 else (5 if diff == 2 else (3 if diff == 3 else 0)))

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
        lost_items = LostItem.query.filter(LostItem.trang_thai != "DA_TIM").all()
        found_items = FoundItem.query.filter(FoundItem.trang_thai != "DA_TRA").all()
    except Exception:
        lost_items, found_items = [], []

    for lost in lost_items:
        for found in found_items:
            cat1 = get_base_category(lost.ten_do, lost.loai_do)
            cat2 = get_base_category(found.ten_do, found.loai_do)

            ten_sim = similarity(lost.ten_do, found.ten_do)

            # LỚP 1: KHÓA TẠO CẶP SAI
            # Nếu 2 món đồ thuộc 2 nhóm danh mục khác nhau hoàn toàn VÀ tên không có từ đồng nghĩa
            if cat1 != cat2 and ten_sim < 0.40:
                continue

            has1 = check_real_image(lost.hinh_anh)
            has2 = check_real_image(found.hinh_anh)

            # LỚP 2: TÍNH ĐIỂM HÌNH ẢNH
            if has1 and has2:
                anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh)
            else:
                anh_score, image_ratio = 0, 0.0

            ten_score = int(ten_sim * 15)

            mau_score = 0
            if lost.mau_sac and found.mau_sac and str(lost.mau_sac).strip().lower() not in ["none", ""] and str(found.mau_sac).strip().lower() not in ["none", ""]:
                mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)
            elif cat1 == cat2 and ten_sim >= 0.70:
                mau_score = 12

            vi_tri_score = 0
            if lost.noi_mat and found.noi_nhat and str(lost.noi_mat).strip().lower() not in ["none", ""] and str(found.noi_nhat).strip().lower() not in ["none", ""]:
                vi_tri = similarity(lost.noi_mat, found.noi_nhat)
                if vi_tri > 0.8:
                    vi_tri_score = 10
                elif vi_tri > 0.5:
                    vi_tri_score = 5

            ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

            if not (has1 and has2):
                anh_score = 0

            # LỚP 3: TÍNH TỔNG ĐIỂM
            score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score

            image_confidence = int(min(100, image_ratio * 100))

            reasons = []
            if has1 and has2 and anh_score >= 20:
                reasons.append("Hình ảnh có điểm tương đồng")
            if ten_score >= 10:
                reasons.append("Tên đồ tương đồng")
            if mau_score >= 10:
                reasons.append("Màu sắc tương đồng")
            if vi_tri_score == 10:
                reasons.append("Vị trí liên quan")
            if ngay_score >= 7:
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

    matches.sort(key=lambda x: x["score"], reverse=True)
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

    if has1 and has2:
        anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh)
    else:
        anh_score, image_ratio = 0, 0.0

    ten_sim = similarity(lost.ten_do, found.ten_do)
    ten_score = int(ten_sim * 15)

    mau_score = 0
    if lost.mau_sac and found.mau_sac and str(lost.mau_sac).strip().lower() not in ["none", ""] and str(found.mau_sac).strip().lower() not in ["none", ""]:
        mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

    vi_tri_score = 0
    if lost.noi_mat and found.noi_nhat and str(lost.noi_mat).strip().lower() not in ["none", ""] and str(found.noi_nhat).strip().lower() not in ["none", ""]:
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
    app.run(host="0.0.0.0", port=5000, debug=True)
