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

# Định nghĩa múi giờ Việt Nam (GMT+7)
VN_TZ = timezone(timedelta(hours=7))

try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"Lỗi OpenCV: {e}")

# --- CẤU HÌNH CLOUDINARY LẤY TỪ ENVIRONMENT VARIABLES TRÊN RENDER ---
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
        print(f"Lỗi khởi tạo DB: {e}")

FEATURE_CACHE = {}

def get_feature_cache(img_path):
    return FEATURE_CACHE.get(img_path) if img_path else None

def save_feature_cache(img_path, features):
    if img_path:
        FEATURE_CACHE[img_path] = features

def extract_foreground_and_mask(img):
    """ Tách vật thể khỏi nền và trả về (vùng_ảnh_cắt, mask_vật_thể) """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding để phát hiện phông nền trắng/trơn
        _, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_area = img.shape[0] * img.shape[1]
            if (w * h) > (img_area * 0.03):
                cropped_img = img[y:y+h, x:x+w]
                cropped_thresh = thresh[y:y+h, x:x+w]
                return cropped_img, cropped_thresh
        return img, thresh
    except Exception:
        return img, None

def calculate_image_features(img_path):
    cached = get_feature_cache(img_path)
    if cached is not None:
        return cached

    try:
        if img_path.startswith("http://") or img_path.startswith("https://"):
            import urllib.request
            req = urllib.request.urlopen(img_path)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        else:
            if not os.path.exists(img_path):
                return None
            img = cv2.imread(img_path)

        if img is None:
            return None

        # 1. Tách nền chuẩn & resize về 300x300
        crop_img, object_mask = extract_foreground_and_mask(img)
        crop_img = cv2.resize(crop_img, (300, 300), interpolation=cv2.INTER_AREA)
        if object_mask is not None:
            object_mask = cv2.resize(object_mask, (300, 300), interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        features = {}

        # 2. ORB Feature Extraction (Giảm nhiễu điểm đặc trưng)
        try:
            orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2, nlevels=8)
            keypoints, descriptors = orb.detectAndCompute(gray, mask=object_mask)
            features["orb_keypoints"] = keypoints
            features["orb_descriptors"] = descriptors
        except Exception:
            features["orb_keypoints"] = []
            features["orb_descriptors"] = None

        # 3. dHash (Chỉ tính trên vùng vật thể đã che phông nền)
        try:
            if object_mask is not None:
                masked_gray = cv2.bitwise_and(gray, gray, mask=object_mask)
            else:
                masked_gray = gray
            resized = cv2.resize(masked_gray, (9, 8), interpolation=cv2.INTER_AREA)
            features["dhash"] = (resized[:, 1:] > resized[:, :-1]).flatten()
        except Exception:
            features["dhash"] = None

        # 4. HSV Color Histogram (Chỉ tính dải màu thuộc vật thể thực sự)
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
    except Exception:
        return None

def image_score(img1_filename, img2_filename, ten_do1="", ten_do2="", noi1="", noi2=""):
    invalid_imgs = ["", "none", "null", "chưa có ảnh", "chua co anh", "no_image.png", "default.jpg"]
    str1 = str(img1_filename).strip().lower() if img1_filename else ""
    str2 = str(img2_filename).strip().lower() if img2_filename else ""

    if not str1 or not str2 or str1 in invalid_imgs or str2 in invalid_imgs:
        return 0, 0.0

    # Cùng link/file -> 50đ tuyệt đối
    name1 = str1.split("/")[-1].split("?")[0]
    name2 = str2.split("/")[-1].split("?")[0]
    if str1 == str2 or name1 == name2:
        return 50, 1.0

    if not str1.startswith("http"):
        p1 = os.path.join(app.config["UPLOAD_FOLDER"], img1_filename)
        if not os.path.exists(p1):
            return 0, 0.0
    if not str2.startswith("http"):
        p2 = os.path.join(app.config["UPLOAD_FOLDER"], img2_filename)
        if not os.path.exists(p2):
            return 0, 0.0

    features1 = calculate_image_features(img1_filename)
    features2 = calculate_image_features(img2_filename)

    if features1 is None or features2 is None:
        return 0, 0.0

    # 1. ORB Feature Matching với Lowe's Ratio Test nghiêm ngặt
    good_count = 0
    try:
        kp1, des1 = features1.get("orb_keypoints", []), features1.get("orb_descriptors")
        kp2, des2 = features2.get("orb_keypoints", []), features2.get("orb_descriptors")
        
        if des1 is not None and des2 is not None and len(kp1) >= 6 and len(kp2) >= 6:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Lọc điểm khớp với tỉ lệ Lowe 0.70 (siết chặt loại bỏ nhiễu phông)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.70 * n.distance:
                        good_matches.append(m)
            good_count = len(good_matches)
    except Exception:
        good_count = 0

    # 2. dHash Structural Matching
    hash_sim = 0.0
    try:
        hash1, hash2 = features1.get("dhash"), features2.get("dhash")
        if hash1 is not None and hash2 is not None:
            hamming_distance = np.count_nonzero(hash1 != hash2)
            hash_sim = 1.0 - (hamming_distance / 64.0)
    except Exception:
        hash_sim = 0.0

    # 3. HSV Color Similarity
    color_corr = 0.0
    try:
        hist1, hist2 = features1.get("hsv_hist"), features2.get("hsv_hist")
        if hist1 is not None and hist2 is not None:
            color_corr = max(0.0, float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)))
    except Exception:
        color_corr = 0.0

    # ==================== NGUYÊN LÝ CHẤM ĐIỂM THUẦN XỬ LÝ ẢNH ====================
    # Nếu số điểm ORB quá ít (< 5 điểm), khẳng định 2 vật thể KHÔNG CÙNG HÌNH DẠNG/MẪU MÃ.
    # Lúc này màu phông nền hay màu sắc tự nhiên không được phép kéo điểm ảnh lên.
    if good_count < 5:
        # Vật thể khác nhau hoàn toàn (ví dụ: Bút bi vs Dây chuyền, Sách vs Laptop)
        orb_points = 0
        hash_points = 5 if hash_sim >= 0.85 else 0
        color_points = int(color_corr * 4)  # Giới hạn màu sắc tối đa 4đ
        total_points = orb_points + hash_points + color_points
        total_points = min(total_points, 8) # KHÓA TRẦN ĐIỂM ẢNH Ở MỨC TỐI ĐA 8/50Đ
    else:
        # Cùng một loại đồ vật (khớp chi tiết ORB thực tế)
        if good_count >= 18:
            orb_points = 38
        elif good_count >= 12:
            orb_points = 30
        elif good_count >= 8:
            orb_points = 22
        else:
            orb_points = 14

        hash_points = 7 if hash_sim >= 0.75 else 0
        color_points = int(color_corr * 5)
        total_points = orb_points + hash_points + color_points
        total_points = min(50, total_points)

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

    invalid_imgs = ["", "none", "null", "chưa có ảnh", "chua co anh", "no_image.png", "default.jpg"]

    for lost in lost_items:
        for found in found_items:
            img1_str = str(lost.hinh_anh).strip().lower() if lost.hinh_anh else ""
            img2_str = str(found.hinh_anh).strip().lower() if found.hinh_anh else ""

            p1_valid = True if img1_str.startswith("http") else os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], img1_str)) if img1_str else False
            p2_valid = True if img2_str.startswith("http") else os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], img2_str)) if img2_str else False

            has_real_images = bool(
                img1_str and img2_str 
                and img1_str not in invalid_imgs 
                and img2_str not in invalid_imgs
                and p1_valid and p2_valid
            )

            if has_real_images:
                anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, lost.ten_do, found.ten_do, lost.noi_mat, found.noi_nhat)
            else:
                anh_score, image_ratio = 0, 0.0

            ten_score = int(similarity(lost.ten_do, found.ten_do) * 15)

            mau_score = 0
            if lost.mau_sac and found.mau_sac and lost.mau_sac.strip() != "" and found.mau_sac.strip() != "":
                mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

            if anh_score == 50 and mau_score < 15:
                mau_score = 15

            vi_tri_score = 0
            if lost.noi_mat and found.noi_nhat:
                vi_tri = similarity(lost.noi_mat, found.noi_nhat)
                if vi_tri > 0.8:
                    vi_tri_score = 10
                elif vi_tri > 0.5:
                    vi_tri_score = 5

            ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

            score = anh_score + ten_score + mau_score + vi_tri_score + ngay_score
            image_confidence = int(min(100, image_ratio * 100))

            reasons = []
            if has_real_images:
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

    invalid_imgs = ["", "none", "null", "chưa có ảnh", "chua co anh", "no_image.png", "default.jpg"]
    img1_str = str(lost.hinh_anh).strip().lower() if lost.hinh_anh else ""
    img2_str = str(found.hinh_anh).strip().lower() if found.hinh_anh else ""

    p1_valid = True if img1_str.startswith("http") else os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], img1_str)) if img1_str else False
    p2_valid = True if img2_str.startswith("http") else os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], img2_str)) if img2_str else False

    has_real_images = bool(
        img1_str and img2_str 
        and img1_str not in invalid_imgs 
        and img2_str not in invalid_imgs
        and p1_valid and p2_valid
    )

    if has_real_images:
        anh_score, image_ratio = image_score(lost.hinh_anh, found.hinh_anh, lost.ten_do, found.ten_do, lost.noi_mat, found.noi_nhat)
    else:
        anh_score, image_ratio = 0, 0.0

    ten_score = int(similarity(lost.ten_do, found.ten_do) * 15)

    mau_score = 0
    if lost.mau_sac and found.mau_sac and lost.mau_sac.strip() != "" and found.mau_sac.strip() != "":
        mau_score = int(similarity(lost.mau_sac, found.mau_sac) * 15)

    if anh_score == 50 and mau_score < 15:
        mau_score = 15

    vi_tri_score = 0
    if lost.noi_mat and found.noi_nhat:
        vi_tri = similarity(lost.noi_mat, found.noi_nhat)
        if vi_tri > 0.8:
            vi_tri_score = 10
        elif vi_tri > 0.5:
            vi_tri_score = 5

    ngay_score = date_score(lost.ngay_mat, found.ngay_nhat)

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
    if has_real_images and image_ratio < 0.20:
        image_warning = "⚠️ Độ tin cậy hình ảnh chưa cao, giáo viên nên kiểm tra kĩ đồ vật thực tế."
    elif not has_real_images:
        image_warning = "⚠️ Bài đăng này không có hình ảnh minh họa, vui lòng xác minh bằng thông tin mô tả."

    return render_template(
        "teacher_check.html",
        lost=lost,
        found=found,
        score=score,
        ket_luan=ket_luan,
        image_warning=image_warning
    )

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
    
    thoi_diem = datetime.now(VN_TZ).replace(tzinfo=None)
    found.thoi_diem_tra = thoi_diem
    lost.thoi_diem_tra = thoi_diem

    db.session.commit()

    return """
    <script>
        alert("Xác nhận đã trả thành công! Bài báo mất đã chuyển sang trạng thái 'Đã tìm thấy'.");
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
