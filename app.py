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

from database import db
db.init_app(app)

from models.lost_item import LostItem
from models.found_item import FoundItem

with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        print(f"Lỗi DB: {e}")

FEATURE_CACHE = {}

def check_real_image(img_path):
    if not img_path:
        return False
    str_path = str(img_path).strip().lower()
    invalid_imgs = ["", "none", "null", "chưa có ảnh", "chua co anh", "no_image.png", "default.jpg"]
    if str_path in invalid_imgs:
        return False
    return True

def get_feature_cache(img_path):
    return FEATURE_CACHE.get(img_path) if img_path else None

def save_feature_cache(img_path, features):
    if img_path:
        FEATURE_CACHE[img_path] = features

def extract_foreground_and_mask(img):
    try:
        if not HAS_OPENCV or img is None:
            return img, None, 1.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratio = 1.0
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_area = img.shape[0] * img.shape[1]
            if h > 0:
                aspect_ratio = float(w) / float(h)
            if (w * h) > (img_area * 0.03):
                return img[y:y+h, x:x+w], thresh[y:y+h, x:x+w], aspect_ratio
        return img, thresh, aspect_ratio
    except Exception:
        return img, None, 1.0

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
            req = urllib.request.urlopen(str(img_path), timeout=4)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            p = os.path.join(app.config["UPLOAD_FOLDER"], str(img_path)) if not os.path.isabs(str(img_path)) else str(img_path)
            if os.path.exists(p):
                img = cv2.imread(p)

        if img is None:
            return None

        crop_img, object_mask, aspect_ratio = extract_foreground_and_mask(img)
        crop_img = cv2.resize(crop_img, (200, 200), interpolation=cv2.INTER_AREA)
        if object_mask is not None:
            object_mask = cv2.resize(object_mask, (200, 200), interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        features = {"aspect_ratio": aspect_ratio}

        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp, des = orb.detectAndCompute(gray, mask=object_mask)
            features["orb_keypoints"] = kp
            features["orb_descriptors"] = des
        except Exception:
            features["orb_keypoints"], features["orb_descriptors"] = [], None

        try:
            if object_mask is not None:
                masked_gray = cv2.bitwise_and(gray, gray, mask=object_mask)
            else:
                masked_gray = gray
            resized = cv2.resize(masked_gray, (9, 8), interpolation=cv2.INTER_AREA)
            features["dhash"] = (resized[:, 1:] > resized[:, :-1]).flatten()
        except Exception:
            features["dhash"] = None

        try:
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], object_mask, [30, 32], [0, 180, 0, 256])
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

def image_score(img1_filename, img2_filename, ten_do1="", ten_do2="", noi1="", noi2=""):
    try:
        # BẮT BỘC KIỂM TRA ẢNH HỢP LỆ TRƯỚC
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
                    if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance:
                        good_count += 1
        except Exception:
            good_count = 0

        hash_sim = 0.0
        try:
            h1, h2 = features1.get("dhash"), features2.get("dhash")
            if h1 is not None and h2 is not None:
                hash_sim = 1.0 - (np.count_nonzero(h1 != h2) / 64.0)
        except Exception:
            hash_sim = 0.0

        color_corr = 0.0
        try:
            hist1, hist2 = features1.get("hsv_hist"), features2.get("hsv_hist")
            if hist1 is not None and hist2 is not None:
                color_corr = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))
        except Exception:
            color_corr = 0.0

        ar1 = features1.get("aspect_ratio", 1.0)
        ar2 = features2.get("aspect_ratio", 1.0)
        shape_diff = abs(ar1 - ar2)

        if good_count >= 12:
            orb_pts = 38
        elif good_count >= 8:
            orb_pts = 30
        elif good_count >= 4:
            orb_pts = 20
        else:
            orb_pts = 8

        if shape_diff > 0.4 and good_count < 6:
            total_points = min(orb_pts + int(color_corr * 3), 8)
        else:
            total_points = orb_pts + (7 if hash_sim >= 0.75 else 0) + int(color_corr * 5)

        total_points = min(50, max(0, total_points))
        return int(total_points), float(total_points / 50.0)
    except Exception as e:
        print(f"Lỗi image_score: {e}")
        return 0, 0.0

def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1, t2 = str(text1).lower().strip(), str(text2).lower().strip()
    if t1 == t2:
        return 1.0
    stopwords = ["sách", "bài tập", "bút", "cái", "chiếc", "vở", "tập"]
    w1 = [w for w in t1.split() if w not in stopwords]
    w2 = [w for w in t2.split() if w not in stopwords]
    
    if t1 in t2 or t2 in t1:
        return 0.85
        
    if w1 and w2 and set(w1).isdisjoint(set(w2)):
        return 0.1

    return SequenceMatcher(None, t1, t2).ratio()

def parse_date_flexible(date_str):
    if not date_str:
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
        
        item = LostItem(
            ten_do=request.form.get("ten_do", ""),
            loai_do=request.form.get("loai_do", ""),
            mau_sac=request.form.get("mau_sac", ""),
            noi_mat=request.form.get("noi_mat", ""),
            ngay_mat=request.form.get("ngay_mat", ""),
            hinh_anh=filename,
            trang_thai="DANG_TIM"
        )
        try:
            db.session.add(item)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
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

        item = FoundItem(
            ten_do=request.form.get("ten_do", ""),
            loai_do=request.form.get("loai_do", ""),
            mau_sac=request.form.get("mau_sac", ""),
            noi_nhat=request.form.get("noi_nhat", ""),
            ngay_nhat=request.form.get("ngay_nhat", ""),
            hinh_anh=filename,
            trang_thai="DANG_LUU_GIU"
        )
        try:
            db.session.add(item)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
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

    try:
        lost_items = lost_query.order_by(LostItem.id.desc()).all()
    except Exception:
        lost_items = []

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

    try:
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
        db.session.rollback()
    return redirect(url_for("list_items"))

@app.route("/edit-lost/<int:item_id>", methods=["GET", "POST"])
@app.route("/edit/<int:item_id>", methods=["GET", "POST"])
def edit_lost(item_id):
    item = LostItem.query.get_or_404(item_id)
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
    item = FoundItem.query.get_or_404(item_id)
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
        db.session.rollback()
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
            has1 = check_real_image(lost.hinh_anh)
            has2 = check_real_image(found.hinh_anh)

            # CHỈ TÍNH ĐIỂM HÌNH ẢNH KHI CẢ 2 BÀI ĐỀU CÓ ẢNH THỰC TẾ
            if has1 and has2:
                anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, lost.ten_do, found.ten_do, lost.noi_mat, found.noi_nhat)
            else:
                anh_score, image_ratio = 0, 0.0

            ten_sim = similarity(lost.ten_do, found.ten_do)
            ten_score = int(ten_sim * 15)

            # NẾU CẢ 2 ĐỀU CÓ ẢNH THỰC TẾ + TÊN GIỐNG NHAU -> BÙ ĐIỂM HÌNH ẢNH NẾU OpenCV SO KHÓ
            if has1 and has2 and ten_sim >= 0.70 and anh_score < 25:
                anh_score = 30
                image_ratio = 0.60

            mau_score = 0
            if lost.mau_sac and found.mau_sac and lost.mau_sac.strip() != "" and found.mau_sac.strip() != "":
                mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

            if has1 and has2 and anh_score >= 35 and mau_score < 15:
                mau_score = 15

            vi_tri_score = 0
            if lost.noi_mat and found.
