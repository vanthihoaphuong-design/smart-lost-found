import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Import database và helper functions từ project
from database import db, LostItem, FoundItem
from test_ai import calculate_image_features, match_items

app = Flask(__name__)
app.secret_key = "smart-lost-found-secret-key"

# Cấu hình thư mục lưu ảnh
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn file tối đa 16MB

# Tự động tạo thư mục uploads nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo CSDL SQLite
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()


def save_and_compress_image(file):
    """Hàm bổ trợ nén ảnh an toàn để tránh sập RAM 512MB trên Render Free Tier"""
    if not file or file.filename == "":
        return ""
    
    try:
        original_name = secure_filename(file.filename)
        filename = f"{uuid.uuid4().hex[:8]}_{original_name}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Nén ảnh về kích thước tối đa 800x800px với chất lượng 75%
        img = Image.open(file)
        img.thumbnail((800, 800))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(file_path, "JPEG", quality=75)

        # Trích xuất đặc trưng hình ảnh an toàn
        try:
            calculate_image_features(file_path)
        except Exception as e:
            print(f"Lỗi tính đặc trưng ảnh: {e}")

        return filename
    except Exception as e:
        print(f"Lỗi xử lý file upload: {e}")
        return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/lost", methods=["GET", "POST"])
def lost():
    if request.method == "POST":
        file = request.files.get("hinh_anh")
        filename = save_and_compress_image(file)

        item = LostItem(
            ten_do=request.form.get("ten_do"),
            loai_do=request.form.get("loai_do"),
            mau_sac=request.form.get("mau_sac"),
            noi_mat=request.form.get("noi_mat"),
            ngay_mat=request.form.get("ngay_mat"),
            hinh_anh=filename,
            trang_thai="DANG_TIM"
        )

        try:
            db.session.add(item)
            db.session.commit()
            flash("Đã đăng bài báo mất thành công!", "success")
        except Exception as e:
            db.session.rollback()
            print(f"Lỗi CSDL lost: {e}")
            flash("Có lỗi xảy ra khi lưu dữ liệu.", "error")

        return redirect(url_for("list_items"))

    return render_template("lost_form.html")


@app.route("/found", methods=["GET", "POST"])
def found():
    if request.method == "POST":
        file = request.files.get("hinh_anh")
        filename = save_and_compress_image(file)

        item = FoundItem(
            ten_do=request.form.get("ten_do"),
            loai_do=request.form.get("loai_do"),
            mau_sac=request.form.get("mau_sac"),
            noi_nhat=request.form.get("noi_nhat"),
            ngay_nhat=request.form.get("ngay_nhat"),
            hinh_anh=filename,
            trang_thai="CHUA_TRA"
        )

        try:
            db.session.add(item)
            db.session.commit()
            flash("Đã đăng bài báo nhặt đồ thành công!", "success")
        except Exception as e:
            db.session.rollback()
            print(f"Lỗi CSDL found: {e}")
            flash("Có lỗi xảy ra khi lưu dữ liệu.", "error")

        return redirect(url_for("list_items"))

    return render_template("found_form.html")


@app.route("/items")
def list_items():
    lost_items = LostItem.query.order_by(LostItem.id.desc()).all()
    found_items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    return render_template("items.html", lost_items=lost_items, found_items=found_items)


@app.route("/match")
def match():
    matches = []
    try:
        matches = match_items()
    except Exception as e:
        print(f"Lỗi đối sánh: {e}")
    return render_template("match.html", matches=matches)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
