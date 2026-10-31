import os
import io
import uuid
import json
from datetime import datetime
from PIL import Image
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher
from flask_sqlalchemy import SQLAlchemy

# Bọc import OpenCV an toàn
try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"Lỗi khởi tạo OpenCV/NumPy: {e}")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "smart_lost_found_secret_key_123")

# 1. Cấu hình thư mục upload ảnh
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 2. Cấu hình PostgreSQL Cloud trên Render
db_url = os.environ.get("DATABASE_URL", "sqlite:///lost_found.db")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# 3. Import các Model Database
from models.lost_item import LostItem
from models.found_item import FoundItem

# 4. Tự động tạo các bảng database
with app.app_context():
    try:
        db.create_all()
        print("Đã khởi tạo thành công các bảng database!")
    except Exception as e:
        print(f"Lỗi khởi tạo database: {e}")

# =========================================================
# BỘ NHỚ ĐỆM CACHE ẢNH TỐI ƯU HIỆU NĂNG
# =========================================================
FEATURE_CACHE = {}

def get_feature_cache(img_path):
    if not img_path:
        return None
    return FEATURE_CACHE.get(img_path)

def save_feature_cache(img_path, features):
    if not img_path:
        return
    FEATURE_CACHE[img_path] = features

def calculate_image_features(img_path):
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    if not os.path.exists(img_path):
        return None

    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        max_size = 400
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        features = {}

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            features["orb_keypoints"] = keypoints
            features["orb_descriptors"] = descriptors
        except Exception:
            features["orb_keypoints"] = []
            features["orb_descriptors"] = None

        try:
            gray_small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_small, (9, 8), interpolation=cv2.INTER_AREA)
            features["dhash"] = (resized[:, 1:] > resized[:, :-1]).flatten()
        except Exception:
            features["dhash"] = None

        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            features["hsv_hist"] = hist
        except Exception:
            features["hsv_hist"] = None

        save_feature_cache(img_path, features)
        return features
    except Exception as e:
        print(f"Lỗi trích xuất đặc trưng ảnh: {e}")
        return None

# =========================================================
# ROUTES GIAO DIỆN VÀ LƯU DỮ LIỆU
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/lost", methods=["GET", "POST"])
def lost():
    if request.method == "POST":
        filename = ""
        file = request.files.get("hinh_anh")
    
        if file and file.filename != "":
            original_name = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(file_path, "JPEG", quality=75)
            except Exception as e:
                print(f"Lỗi nén ảnh lost: {e}")
                file.save(file_path)

            try:
                calculate_image_features(file_path)
            except Exception as e:
                print(f"Lỗi tính đặc trưng lost: {e}")

        ten_do = request.form.get("ten_do", "")
        loai_do = request.form.get("loai_do", "")
        mau_sac = request.form.get("mau_sac", "")
        noi_mat = request.form.get("noi_mat", "")
        ngay_mat = request.form.get("ngay_mat", "")

        # Khởi tạo đối tượng LostItem linh hoạt mọi tham số
        item = LostItem.__new__(LostItem)
        item.__init__()

        for attr, val in [
            ('ten_do', ten_do), ('name', ten_do), ('title', ten_do),
            ('loai_do', loai_do), ('category', loai_do),
            ('mau_sac', mau_sac), ('color', mau_sac), ('description', f"Màu sắc: {mau_sac}"),
            ('noi_mat', noi_mat), ('location', noi_mat),
            ('ngay_mat', ngay_mat), ('date', ngay_mat),
            ('hinh_anh', filename), ('image_path', f"uploads/{filename}" if filename else None), ('image', filename),
            ('trang_thai', 'DANG_TIM'), ('status', 'DANG_TIM')
        ]:
            setattr(item, attr, val)

        try:
            db.session.add(item)
            db.session.commit()
            print(">>> ĐÃ LƯU THÀNH CÔNG BÁO MẤT!")
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI COMMIT BÁO MẤT: {e}")

        return redirect(url_for("list_items"))

    return render_template("lost_form.html")

@app.route("/found", methods=["GET", "POST"])
def found():
    if request.method == "POST":
        filename = ""
        file = request.files.get("hinh_anh")

        if file and file.filename != "":
            original_name = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(file_path, "JPEG", quality=75)
            except Exception as e:
                print(f"Lỗi nén ảnh found: {e}")
                file.save(file_path)

            try:
                calculate_image_features(file_path)
            except Exception as e:
                print(f"Lỗi tính đặc trưng found: {e}")

        ten_do = request.form.get("ten_do", "")
        loai_do = request.form.get("loai_do", "")
        mau_sac = request.form.get("mau_sac", "")
        noi_nhat = request.form.get("noi_nhat", "")
        ngay_nhat = request.form.get("ngay_nhat", "")

        # Khởi tạo đối tượng FoundItem linh hoạt mọi tham số
        item = FoundItem.__new__(FoundItem)
        item.__init__()

        for attr, val in [
            ('ten_do', ten_do), ('name', ten_do), ('title', ten_do),
            ('loai_do', loai_do), ('category', loai_do),
            ('mau_sac', mau_sac), ('color', mau_sac), ('description', f"Màu sắc: {mau_sac}"),
            ('noi_nhat', noi_nhat), ('noi_mat', noi_nhat), ('location', noi_nhat),
            ('ngay_nhat', ngay_nhat), ('ngay_mat', ngay_nhat), ('date', ngay_nhat),
            ('hinh_anh', filename), ('image_path', f"uploads/{filename}" if filename else None), ('image', filename),
            ('trang_thai', 'DANG_LUU_GIU'), ('status', 'DANG_LUU_GIU')
        ]:
            setattr(item, attr, val)

        try:
            db.session.add(item)
            db.session.commit()
            print(">>> ĐÃ LƯU THÀNH CÔNG BÁO NHẶT!")
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI COMMIT BÁO NHẶT: {e}")

        return redirect(url_for("found_list"))

    return render_template("found_form.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
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
        if hasattr(LostItem, 'ten_do'):
            lost_query = lost_query.filter(LostItem.ten_do.ilike(f"%{keyword}%"))
        elif hasattr(LostItem, 'name'):
            lost_query = lost_query.filter(LostItem.name.ilike(f"%{keyword}%"))
    if category:
        if hasattr(LostItem, 'loai_do'):
            lost_query = lost_query.filter(LostItem.loai_do == category)
        elif hasattr(LostItem, 'category'):
            lost_query = lost_query.filter(LostItem.category == category)
    if location:
        if hasattr(LostItem, 'noi_mat'):
            lost_query = lost_query.filter(LostItem.noi_mat == location)
        elif hasattr(LostItem, 'location'):
            lost_query = lost_query.filter(LostItem.location == location)
    if color:
        if hasattr(LostItem, 'mau_sac'):
            lost_query = lost_query.filter(LostItem.mau_sac.ilike(f"%{color}%"))

    if status in ("DANG_TIM", "DA_TIM", "Chưa tìm thấy", "Đã tìm thấy"):
        if hasattr(LostItem, 'trang_thai'):
            lost_query = lost_query.filter(LostItem.trang_thai == status)
    elif status in ("DANG_LUU_GIU", "DA_TRA", "Đang lưu giữ", "Đã hoàn trả"):
        lost_query = lost_query.filter(LostItem.id == -1)

    try:
        lost_items = lost_query.order_by(LostItem.id.desc()).all()
    except Exception:
        lost_items = []

    found_query = FoundItem.query
    if keyword:
        if hasattr(FoundItem, 'ten_do'):
            found_query = found_query.filter(FoundItem.ten_do.ilike(f"%{keyword}%"))
        elif hasattr(FoundItem, 'name'):
            found_query = found_query.filter(FoundItem.name.ilike(f"%{keyword}%"))
    if category:
        if hasattr(FoundItem, 'loai_do'):
            found_query = found_query.filter(FoundItem.loai_do == category)
        elif hasattr(FoundItem, 'category'):
            found_query = found_query.filter(FoundItem.category == category)
    if location:
        if hasattr(FoundItem, 'noi_nhat'):
            found_query = found_query.filter(FoundItem.noi_nhat == location)
        elif hasattr(FoundItem, 'location'):
            found_query = found_query.filter(FoundItem.location == location)
    if color:
        if hasattr(FoundItem, 'mau_sac'):
            found_query = found_query.filter(FoundItem.mau_sac.ilike(f"%{color}%"))

    if status in ("DANG_LUU_GIU", "DA_TRA", "Đang lưu giữ", "Đã hoàn trả"):
        if hasattr(FoundItem, 'trang_thai'):
            found_query = found_query.filter(FoundItem.trang_thai == status)
    elif status in ("DANG_TIM", "DA_TIM", "Chưa tìm thấy", "Đã tìm thấy"):
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
    item = LostItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
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
            original_name = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(filepath, "JPEG", quality=75)
            except Exception:
                file.save(filepath)

            item.hinh_anh = filename
            try:
                calculate_image_features(filepath)
            except Exception:
                pass
            
        db.session.commit()
        return redirect(url_for("list_items"))
        
    return render_template("edit_item.html", item=item, is_lost=True)

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
            original_name = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(filepath, "JPEG", quality=75)
            except Exception:
                file.save(filepath)

            item.hinh_anh = filename
            try:
                calculate_image_features(filepath)
            except Exception:
                pass
            
        db.session.commit()
        return redirect(url_for("found_list"))
        
    return render_template("edit_item.html", item=item, is_lost=False)

@app.route("/dashboard")
def dashboard():
    try:
        tong_so = LostItem.query.count()
        all_items = LostItem.query.all()
    except Exception:
        tong_so = 0
        all_items = []

    CATEGORIES_MAP = {
        "Sách vở / Dụng cụ học tập": [
            "Sách vở / Dụng cụ học tập", "Sách vở", "Sách", "Vở", "Tập", 
            "Bút", "Thước", "Bóp viết", "Sách Tin học", "Sách tin 11", "Dụng cụ học tập"
        ],
        "Thiết bị điện tử (Laptop, Điện thoại...)": [
            "Thiết bị điện tử (Laptop, Điện thoại...)", "Thiết bị điện tử", 
            "Laptop", "Điện thoại", "Máy tính Casio", "Casio", "mouse", "máy tính", "Chuột"
        ],
        "Trang phục / Phụ kiện (Áo khoác, Ví...)": [
            "Trang phục / Phụ kiện (Áo khoác, Ví...)", "Trang phục / Phụ kiện", 
            "Áo khoác", "Balo", "Mũ", "Ví", "Vali", "Cặp", "Nón"
        ],
        "Giấy tờ / Thẻ học sinh": [
            "Giấy tờ / Thẻ học sinh", "Giấy tờ", "Thẻ học sinh", "Thẻ HS", "Căn cước", "Thẻ"
        ],
        "Khác": [
            "Khác", "Bình nước", "Không xác định", "vase", "Khu vực khác"
        ]
    }

    category_counts = {cat: 0 for cat in CATEGORIES_MAP.keys()}

    for item in all_items:
        raw_cat = str(getattr(item, 'loai_do', getattr(item, 'category', 'Khác'))).strip()
        mapped = False
        
        for new_cat, old_list in CATEGORIES_MAP.items():
            if any(old.lower() in raw_cat.lower() or raw_cat.lower() in old.lower() for old in old_list):
                category_counts[new_cat] += 1
                mapped = True
                break
                
        if not mapped:
            category_counts["Khác"] += 1

    thong_ke = [(cat, count) for cat, count in category_counts.items()]

    return render_template(
        "dashboard.html",
        tong_so=tong_so,
        thong_ke=thong_ke
    )

@app.route("/heatmap")
def heatmap():
    areas = [
        "Sân trường", "Nhà xe", "Căn tin", "Thư viện", "Sân bóng",
        "Phòng STEM", "Phòng Tin học 1", "Phòng Tin học 2",
        "Phòng Văn", "Phòng Sử - Địa", "Phòng Hội trường",
        "Phòng Đoàn - Đội", "Khu Văn phòng", "Phòng Tâm lý",
        "6A1", "6A2", "6A3",
        "7A1", "7A2", "7A3",
        "8A1", "8A2", "8A3",
        "9A1", "9A2", "9A3",
        "10A1", "10A2", "10A3",
        "11A1", "11A2",
        "12A1", "12A2"
    ]

    try:
        db_data = db.session.query(
            LostItem.noi_mat,
            func.count(LostItem.id)
        ).group_by(LostItem.noi_mat).all()
        count_dict = dict(db_data)
    except Exception:
        count_dict = {}

    data = [(area, count_dict.get(area, 0)) for area in areas]

    return render_template("heatmap.html", data=data)

@app.route("/found-return/<int:id>")
def found_return(id):
    item = FoundItem.query.get_or_404(id)
    item.trang_thai = "DA_TRA"
    if hasattr(item, 'thoi_diem_tra'):
        item.thoi_diem_tra = datetime.now()
    db.session.commit()
    return redirect(url_for("found_list"))

# =========================================================
# THUẬT TOÁN ĐỐI SÁNH TRÍ TUỆ NHÂN TẠO AI
# =========================================================

def similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    t1 = str(text1).lower().strip()
    t2 = str(text2).lower().strip()

    if t1 in t2 or t2 in t1:
        return 1.0

    return SequenceMatcher(None, t1, t2).ratio()

def date_score(date1, date2):
    try:
        d1 = datetime.strptime(str(date1), "%Y-%m-%d")
        d2 = datetime.strptime(str(date2), "%Y-%m-%d")
        diff = abs((d1 - d2).days)

        if diff == 0:
            return 10
        elif diff == 1:
            return 7
        elif diff == 2:
            return 3
        else:
            return 0
    except Exception:
        return 0

def image_score(img1_filename, img2_filename):
    if not img1_filename or not img2_filename:
        return 0, 0.0

    if img1_filename == img2_filename:
        return 50, 1.0

    img1_path = os.path.join(app.config["UPLOAD_FOLDER"], img1_filename)
    img2_path = os.path.join(app.config["UPLOAD_FOLDER"], img2_filename)

    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return 0, 0.0

    features1 = calculate_image_features(img1_path)
    features2 = calculate_image_features(img2_path)

    if features1 is None or features2 is None:
        return 0, 0.0

    hash_points = 0
    hash_sim = 0.0
    try:
        hash1, hash2 = features1.get("dhash"), features2.get("dhash")
        if hash1 is not None and hash2 is not None:
            hamming_distance = np.count_nonzero(hash1 != hash2)
            hash_sim = 1.0 - (hamming_distance / 64.0)

            if hamming_distance <= 3 or hash_sim >= 0.95:
                return 50, 1.0

            if hash_sim >= 0.85:
                hash_points = 15
            elif hash_sim >= 0.70:
                hash_points = 9
            elif hash_sim >= 0.50:
                hash_points = 3
    except Exception:
        hash_points = 0

    orb_points = 0
    try:
        kp1, des1 = features1.get("orb_keypoints", []), features1.get("orb_descriptors")
        kp2, des2 = features2.get("orb_keypoints", []), features2.get("orb_descriptors")

        if des1 is not None and des2 is not None and len(kp1) >= 5 and len(kp2) >= 5:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.72 * n.distance:
                    good_matches.append(m)

            good_count = len(good_matches)
            inliers = 0
            inlier_ratio = 0.0

            if good_count >= 5:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                try:
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if mask is not None:
                        inliers = int(np.sum(mask.ravel()))
                        inlier_ratio = inliers / good_count if good_count > 0 else 0.0
                except Exception:
                    pass

            good_score = 10 if good_count >= 30 else (7 if good_count >= 15 else (3 if good_count >= 8 else 0))
            inlier_score = 10 if inliers >= 20 else (7 if inliers >= 10 else (3 if inliers >= 5 else 0))
            ratio_score = 5 if inlier_ratio >= 0.60 else (3 if inlier_ratio >= 0.35 else 0)

            orb_points = good_score + inlier_score + ratio_score
    except Exception:
        orb_points = 0

    color_points = 0
    try:
        hist1, hist2 = features1.get("hsv_hist"), features2.get("hsv_hist")
        if hist1 is not None and hist2 is not None:
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            correlation = max(0.0, min(1.0, (correlation + 1) / 2))

            if correlation >= 0.85:
                color_points = 10
            elif correlation >= 0.70:
                color_points = 6
            elif correlation >= 0.50:
                color_points = 2
    except Exception:
        color_points = 0

    total_points = min(50, orb_points + hash_points + color_points)
    image_ratio = total_points / 50.0

    return int(total_points), float(image_ratio)

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
            anh_score, image_ratio = image_score(getattr(lost, 'hinh_anh', None), getattr(found, 'hinh_anh', None))
            ten_score = int(similarity(getattr(lost, 'ten_do', ''), getattr(found, 'ten_do', '')) * 15)
            mau_score = int(similarity(getattr(lost, 'mau_sac', ''), getattr(found, 'mau_sac', '')) * 15)

            vi_tri_score = 0
            noi_mat = getattr(lost, 'noi_mat', getattr(lost, 'location', ''))
            noi_nhat = getattr(found, 'noi_nhat', getattr(found, 'location', ''))
            if noi_mat and noi_nhat:
                vi_tri = similarity(noi_mat, noi_nhat)
                if vi_tri > 0.8:
                    vi_tri_score = 10
                elif vi_tri > 0.5:
                    vi_tri_score = 5

            ngay_mat = getattr(lost, 'ngay_mat', getattr(lost, 'date', ''))
            ngay_nhat = getattr(found, 'ngay_nhat', getattr(found, 'date', ''))
            ngay_score = date_score(ngay_mat, ngay_nhat)

            score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score
            image_confidence = int(min(100, image_ratio * 100))

            reasons = []
            if anh_score >= 35:
                reasons.append("Hình ảnh rất giống")
            elif anh_score >= 15:
                reasons.append("Hình ảnh có điểm tương đồng")

            if ten_score >= 10:
                reasons.append("Tên đồ tương đồng")
            if mau_score >= 10:
                reasons.append("Màu sắc tương đồng")
            if vi_tri_score == 10:
                reasons.append("Vị trí liên quan")
            if ngay_score >= 7:
                reasons.append("Thời gian khớp/gần nhau")

            if score >= 80:
                ket_luan = "🟢 Khả năng trùng khớp cao"
            elif score >= 60:
                ket_luan = "🟡 Có khả năng trùng khớp"
            elif score >= 40:
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
    lost = LostItem.query.get_or_404(lost_id)
    found = FoundItem.query.get_or_404(found_id)

    anh_score, image_ratio = image_score(getattr(lost, 'hinh_anh', None), getattr(found, 'hinh_anh', None))
    ten_score = int(similarity(getattr(lost, 'ten_do', ''), getattr(found, 'ten_do', '')) * 15)

    mau_score = 0
    if getattr(lost, 'mau_sac', None) and getattr(found, 'mau_sac', None):
        mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

    vi_tri_score = 0
    noi_mat = getattr(lost, 'noi_mat', getattr(lost, 'location', ''))
    noi_nhat = getattr(found, 'noi_nhat', getattr(found, 'location', ''))
    if noi_mat and noi_nhat:
        vi_tri = similarity(noi_mat, noi_nhat)
        if vi_tri > 0.8:
            vi_tri_score = 10
        elif vi_tri > 0.5:
            vi_tri_score = 5

    ngay_mat = getattr(lost, 'ngay_mat', getattr(lost, 'date', ''))
    ngay_nhat = getattr(found, 'ngay_nhat', getattr(found, 'date', ''))
    ngay_score = date_score(ngay_mat, ngay_nhat)

    score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score

    if score >= 80:
        ket_luan = "🟢 Khả năng trùng khớp cao"
    elif score >= 60:
        ket_luan = "🟡 Có khả năng trùng khớp"
    elif score >= 40:
        ket_luan = "🟠 Có một số điểm tương đồng"
    else:
        ket_luan = "🔴 Khả năng trùng khớp thấp"

    image_warning = None
    if image_ratio < 0.20:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."

    return render_template(
        "teacher_check.html",
        lost=lost,
        found=found,
        score=score,
        ket_luan=ket_luan,
        image_warning=image_warning
    )

# =========================================================
# QUẢN LÝ MÃ PIN & BÁO CÁO EXCEL
# =========================================================

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

    lost = LostItem.query.get_or_404(lost_id)
    found = FoundItem.query.get_or_404(found_id)

    found.trang_thai = "DA_TRA"
    lost.trang_thai = "DA_TIM"
    
    thoi_diem = datetime.now()
    if hasattr(found, 'thoi_diem_tra'):
        found.thoi_diem_tra = thoi_diem
    if hasattr(lost, 'thoi_diem_tra'):
        lost.thoi_diem_tra = thoi_diem

    db.session.commit()

    return """
    <script>
        alert("Xac nhan hoan tra thanh cong! Bai bao mat da chuyen sang trang thai 'Da tim thay'.");
        window.location.href = "/lost-list";
    </script>
    """

@app.route("/export-excel", methods=["GET"])
def export_excel():
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
        "Tên đồ vật": getattr(item, 'ten_do', getattr(item, 'name', '')),
        "Loại đồ": getattr(item, 'loai_do', getattr(item, 'category', '')),
        "Màu sắc": getattr(item, 'mau_sac', getattr(item, 'color', '')),
        "Nơi mất": getattr(item, 'noi_mat', getattr(item, 'location', '')),
        "Ngày mất": getattr(item, 'ngay_mat', getattr(item, 'date', '')),
        "Trạng thái": "Đã tìm thấy" if getattr(item, 'trang_thai', '') in ["DA_TIM", "Đã tìm thấy"] else "Đang tìm"
    } for i, item in enumerate(lost_items)]

    found_data = [{
        "STT": i + 1,
        "Tên đồ vật": getattr(item, 'ten_do', getattr(item, 'name', '')),
        "Loại đồ": getattr(item, 'loai_do', getattr(item, 'category', '')),
        "Màu sắc": getattr(item, 'mau_sac', getattr(item, 'color', '')),
        "Nơi nhặt": getattr(item, 'noi_nhat', getattr(item, 'location', '')),
        "Ngày nhặt": getattr(item, 'ngay_nhat', getattr(item, 'date', '')),
        "Trạng thái": "Đã hoàn trả" if getattr(item, 'trang_thai', '') in ["DA_TRA", "Đã hoàn trả"] else "Đang lưu giữ"
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
