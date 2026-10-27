import os
import io
from PIL import Image
import pandas as pd
from flask import send_file, flash, redirect, url_for
import uuid
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from sqlalchemy import func
from difflib import SequenceMatcher
from datetime import datetime
import cv2
import numpy as np
import json
from database import db
from models.lost_item import LostItem
from models.found_item import FoundItem
from ai.detect import detect_object
from ultralytics import YOLO
from flask import jsonify


model = YOLO("yolov8n.pt")

# =========================================================
# BỘ NHỚ ĐỆM (CACHE) ĐẶC TRƯNG ẢNH TỐI ƯU HIỆU NĂNG
# =========================================================
IMAGE_CACHE = {}
FEATURE_CACHE = {}

def get_feature_cache(img_path):
    if not img_path:
        return None
    return FEATURE_CACHE.get(img_path)

def save_feature_cache(img_path, features):
    if not img_path:
        return
    FEATURE_CACHE[img_path] = features

def crop_object(img_path, yolo_model):
    img = cv2.imread(img_path)
    if img is None:
        return None

    results = yolo_model(img, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return img

    boxes = results[0].boxes
    best_box = None
    best_conf = 0

    for box in boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return img

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    h, w = img.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img

    return crop

def calculate_image_features(img_path):
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    img = crop_object(img_path, model)
    if img is None:
        return None

    # Resize tối ưu kích thước ảnh để tính toán cực nhanh (max 400px)
    max_size = 400
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    features = {}

    # 1. ORB (Giảm nfeatures xuống 500 giúp tăng tốc gấp 5 lần)
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

    # 2. dHASH
    try:
        gray_small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_small, (9, 8), interpolation=cv2.INTER_AREA)
        features["dhash"] = (resized[:, 1:] > resized[:, :-1]).flatten()
    except Exception:
        features["dhash"] = None

    # 3. HSV Histogram
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        features["hsv_hist"] = hist
    except Exception:
        features["hsv_hist"] = None

    save_feature_cache(img_path, features)
    return features


app = Flask(__name__)
app.secret_key = "smart_lost_found_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///lost_found.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db.init_app(app)
with app.app_context():
    db.create_all()


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
        
            # Nén ảnh để tránh tràn RAM 512MB trên Render
            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(file_path, "JPEG", quality=75)
            except Exception as e:
                print(f"Lỗi nén ảnh lost: {e}")
                file.save(file_path)

            # Trích xuất cache đặc trưng an toàn
            try:
                calculate_image_features(file_path)
            except Exception as e:
                print(f"Lỗi calculate_image_features lost: {e}")

        ai_name = "Không xác định"
        score = 0

        if filename != "":
            try:
                ai_name, score = detect_object(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            except Exception as e:
                print(f"Lỗi detect_object lost: {e}")

        item = LostItem(
            ten_do=request.form.get("ten_do"),
            loai_do=request.form.get("loai_do"),
            mau_sac=request.form.get("mau_sac"),
            noi_mat=request.form.get("noi_mat"),
            ngay_mat=request.form.get("ngay_mat"),
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
            original_name = secure_filename(file.filename)
            filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Nén ảnh để tránh tràn RAM 512MB trên Render
            try:
                img = Image.open(file)
                img.thumbnail((800, 800))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(file_path, "JPEG", quality=75)
            except Exception as e:
                print(f"Lỗi nén ảnh found: {e}")
                file.save(file_path)

            # Trích xuất cache đặc trưng an toàn
            try:
                calculate_image_features(file_path)
            except Exception as e:
                print(f"Lỗi calculate_image_features found: {e}")

        ai_name = "Không xác định"
        score = 0

        if filename != "":
            try:
                ai_name, score = detect_object(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            except Exception as e:
                print(f"Lỗi detect_object found: {e}")

        item = FoundItem(
            ten_do=request.form.get("ten_do"),
            loai_do=request.form.get("loai_do"),
            mau_sac=request.form.get("mau_sac"),
            noi_nhat=request.form.get("noi_nhat"),
            ngay_nhat=request.form.get("ngay_nhat"),
            hinh_anh=filename,
            trang_thai="DANG_LUU_GIU"
        )

        db.session.add(item)
        db.session.commit()

        return redirect(url_for("found_list"))

    return render_template("found_form.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/lost-list")
def list_items():
    items = LostItem.query.order_by(LostItem.id.desc()).all()
    return render_template("list_items.html", items=items)


@app.route("/found-list")
def found_list():
    items = FoundItem.query.order_by(FoundItem.id.desc()).all()
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
        lost_query = lost_query.filter(LostItem.ten_do.contains(keyword))
    if category:
        lost_query = lost_query.filter(LostItem.loai_do == category)
    if location:
        lost_query = lost_query.filter(LostItem.noi_mat == location)
    if color:
        lost_query = lost_query.filter(LostItem.mau_sac.contains(color))

    if status in ("DANG_TIM", "DA_TIM"):
        lost_query = lost_query.filter(LostItem.trang_thai == status)
    elif status in ("DANG_LUU_GIU", "DA_TRA"):
        lost_query = lost_query.filter(LostItem.id == -1)

    lost_items = lost_query.order_by(LostItem.id.desc()).all()

    found_query = FoundItem.query
    if keyword:
        found_query = found_query.filter(FoundItem.ten_do.contains(keyword))
    if category:
        found_query = found_query.filter(FoundItem.loai_do == category)
    if location:
        found_query = found_query.filter(FoundItem.noi_nhat == location)
    if color:
        found_query = found_query.filter(FoundItem.mau_sac.contains(color))

    if status in ("DANG_LUU_GIU", "DA_TRA"):
        found_query = found_query.filter(FoundItem.trang_thai == status)
    elif status in ("DANG_TIM", "DA_TIM"):
        found_query = found_query.filter(FoundItem.id == -1)

    found_items = found_query.order_by(FoundItem.id.desc()).all()

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


# =========================================================
# ROUTE QUẢN LÝ / CHỈNH SỬA / XÓA ĐỒ THẤT LẠC
# =========================================================
@app.route("/delete-lost/<int:item_id>")
@app.route("/delete/<int:item_id>")
def delete_item(item_id):
    item = LostItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for("list_items"))


# =========================================================
# ROUTE CHỈNH SỬA ĐỒ MẤT VÀ ĐỒ NHẶT (DÙNG CHUNG 1 TEMPLATE)
# =========================================================
@app.route("/edit-lost/<int:item_id>", methods=["GET", "POST"])
@app.route("/edit/<int:item_id>", methods=["GET", "POST"])
def edit_lost(item_id):
    item = LostItem.query.get_or_404(item_id)
    if request.method == "POST":
        item.ten_do = request.form.get("ten_do")
        item.loai_do = request.form.get("lo
