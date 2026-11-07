import os
import io
import uuid
import json
from datetime import datetime
from PIL import Image
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher

import cloudinary
import cloudinary.uploader
from datetime import datetime, timezone, timedelta

# Múi giờ Việt Nam
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

def get_feature_cache(img_path):
    return FEATURE_CACHE.get(img_path) if img_path else None

def save_feature_cache(img_path, features):
    if img_path:
        FEATURE_CACHE[img_path] = features

def extract_foreground_and_mask(img):
    try:
        if not HAS_OPENCV or img is None:
            return img, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_area = img.shape[0] * img.shape[1]
            if (w * h) > (img_area * 0.03):
                return img[y:y+h, x:x+w], thresh[y:y+h, x:x+w]
        return img, thresh
    except Exception:
        return img, None

def calculate_image_features(img_path):
    if not HAS_OPENCV or not img_path:
        return None
    
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        img = None
        if img_path.startswith("http://") or img_path.startswith("https://"):
            import urllib.request
            req = urllib.request.urlopen(img_path, timeout=5)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            p = os.path.join(app.config["UPLOAD_FOLDER"], img_path) if not os.path.isabs(img_path) else img_path
            if os.path.exists(p):
                img = cv2.imread(p)

        if img is None:
            return None

        crop_img, object_mask = extract_foreground_and_mask(img)
        crop_img = cv2.resize(crop_img, (250, 250), interpolation=cv2.INTER_AREA)
        if object_mask is not None:
            object_mask = cv2.resize(object_mask, (250, 250), interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        features = {}

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
        invalid_imgs = ["", "none", "null", "chưa có ảnh", "chua co anh", "no_image.png", "default.jpg"]
        str1 = str(img1_filename).strip().lower() if img1_filename else ""
        str2 = str(img2_filename).strip().lower() if img2_filename else ""

        if not str1 or not str2 or str1 in invalid_imgs or str2 in invalid_imgs:
            return 0, 0.0

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
            if des1 is not None and des2 is not None and len(kp1) >= 5 and len(kp2) >= 5:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(des1, des2, k=2)
                for m_pair in matches:
                    if len(m_pair) == 2 and m_pair[0].distance < 0.70 * m_pair[1].distance:
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

        if good_count < 5:
            total_points = min(int(color_corr * 4) + (5 if hash_sim >= 0.85 else 0), 8)
        else:
            orb_pts = 38 if good_count >= 18 else (30 if good_count >= 12 else (22 if good_count >= 8 else 14))
            total_points = min(50, orb_pts + (7 if hash_sim >= 0.75 else 0) + int(color_corr * 5))

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
    if w1 and w2 and set(w1).isdisjoint(set(w2)):
        return 0.1
    if t1 in t2 or t2 in t1:
        return 0.85
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
        db.session.add(item)
        db.session.commit()
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
        db.session.add(item)
        db.session.commit()
        return redirect(url_for("found_list"))
    return render_template("found_form.html")

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
            anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, lost.ten_do, found.ten_do)
            ten_score = int(similarity(lost.ten_do, found.ten_do) * 15)
            
            mau_score = 0
            if lost.mau_sac and found.mau_sac and lost.mau_sac.strip() != "":
                mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

            vi_tri_score = 0
            if lost.noi_mat and found.noi_nhat:
                vi_tri = similarity(lost.noi_mat, found.noi_nhat)
                vi_tri_score = 10 if vi_tri > 0.8 else (5 if vi_tri > 0.5 else 0)

            ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)
            score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score

            ket_luan = "🟢 Khả năng trùng khớp cao" if score >= 80 else ("🟡 Có khả năng trùng khớp" if score >= 60 else ("🟠 Có một số điểm tương đồng" if score >= 40 else "🔴 Khả năng trùng khớp thấp"))

            matches.append({
                "lost": lost, "found": found, "score": score, "ket_luan": ket_luan,
                "score_image": anh_score, "score_name": ten_score, "score_color": mau_score,
                "score_location": vi_tri_score, "score_date": ngay_score
            })

    matches.sort(key=lambda x: x["score"], reverse=True)
    return render_template("match.html", matches=matches)

@app.route("/lost-list")
def list_items():
    items = LostItem.query.order_by(LostItem.id.desc()).all()
    return render_template("list_items.html", items=items)

@app.route("/found-list")
def found_list():
    items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    return render_template("found_list.html", items=items)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
