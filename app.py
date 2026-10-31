import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super-secret-key-123')

# 1. Cấu hình upload thư mục ảnh
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2. Cấu hình kết nối Cloud PostgreSQL cho Render
db_url = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 3. Import các Model Database
from models.lost_item import LostItem
from models.found_item import FoundItem

# 4. Tự động khởi tạo đầy đủ các bảng trên Cloud PostgreSQL
with app.app_context():
    try:
        db.create_all()
        print("Đã tạo bảng database thành công!")
    except Exception as e:
        print(f"Lỗi tạo bảng: {e}")

# ==========================================
# CÁC ROUTE XỬ LÝ GIAO DIỆN VÀ DỮ LIỆU
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lost', methods=['GET', 'POST'])
def lost_item():
    if request.method == 'POST':
        try:
            ten_do = request.form.get('ten_do') or ''
            loai_do = request.form.get('loai_do') or ''
            mau_sac = request.form.get('mau_sac') or ''
            noi_mat = request.form.get('noi_mat') or ''
            ngay_mat = request.form.get('ngay_mat') or ''
            
            file = request.files.get('hinh_anh')
            image_path = None
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_path = f"uploads/{filename}"

            new_item = LostItem()
            # Gán linh hoạt cho mọi kiểu thiết kế Model (Tiếng Việt hoặc Tiếng Anh)
            for attr, val in [
                ('ten_do', ten_do), ('name', ten_do), ('title', ten_do),
                ('loai_do', loai_do), ('category', loai_do),
                ('mau_sac', mau_sac), ('color', mau_sac), ('description', f"Màu sắc: {mau_sac}"),
                ('noi_mat', noi_mat), ('location', noi_mat),
                ('ngay_mat', ngay_mat), ('date', ngay_mat),
                ('hinh_anh', image_path), ('image_path', image_path), ('image', image_path)
            ]:
                if hasattr(new_item, attr):
                    setattr(new_item, attr, val)

            db.session.add(new_item)
            db.session.commit()
            print(">>> ĐÃ LƯU THÀNH CÔNG BÀI BÁO MẤT!")
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI LƯU BÁO MẤT: {e}")
        
        return redirect(url_for('lost_list'))
            
    return render_template('lost_form.html')

@app.route('/found', methods=['GET', 'POST'])
def found_item():
    if request.method == 'POST':
        try:
            ten_do = request.form.get('ten_do') or ''
            loai_do = request.form.get('loai_do') or ''
            mau_sac = request.form.get('mau_sac') or ''
            noi_nhat = request.form.get('noi_nhat') or ''
            ngay_nhat = request.form.get('ngay_nhat') or ''
            
            file = request.files.get('hinh_anh')
            image_path = None
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_path = f"uploads/{filename}"

            new_item = FoundItem()
            for attr, val in [
                ('ten_do', ten_do), ('name', ten_do), ('title', ten_do),
                ('loai_do', loai_do), ('category', loai_do),
                ('mau_sac', mau_sac), ('color', mau_sac), ('description', f"Màu sắc: {mau_sac}"),
                ('noi_nhat', noi_nhat), ('location', noi_nhat),
                ('ngay_nhat', ngay_nhat), ('date', ngay_nhat),
                ('hinh_anh', image_path), ('image_path', image_path), ('image', image_path)
            ]:
                if hasattr(new_item, attr):
                    setattr(new_item, attr, val)

            db.session.add(new_item)
            db.session.commit()
            print(">>> ĐÃ LƯU THÀNH CÔNG BÀI BÁO NHẶT!")
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI LƯU BÁO NHẶT: {e}")

        return redirect(url_for('found_list'))

    return render_template('found_form.html')

@app.route('/lost-list')
def lost_list():
    try:
        items = LostItem.query.order_by(LostItem.id.desc()).all()
    except Exception as e:
        items = []
    return render_template('list_items.html', items=items, type='lost')

@app.route('/found-list')
def found_list():
    try:
        items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    except Exception as e:
        items = []
    try:
        return render_template('found_list.html', items=items, type='found')
    except:
        return render_template('list_items.html', items=items, type='found')

@app.route('/match')
def match_items():
    try:
        lost_items = LostItem.query.all()
        found_items = FoundItem.query.all()
    except Exception as e:
        lost_items, found_items = [], []
    return render_template('match.html', lost_items=lost_items, found_items=found_items)

@app.route('/dashboard')
def dashboard():
    try:
        total_lost = LostItem.query.count()
        total_found = FoundItem.query.count()
    except Exception as e:
        total_lost, total_found = 0, 0
    return render_template('dashboard.html', total_lost=total_lost, total_found=total_found)

@app.route('/heatmap')
def heatmap():
    try:
        lost_items = LostItem.query.all()
        found_items = FoundItem.query.all()
    except Exception as e:
        lost_items, found_items = [], []
    return render_template('heatmap.html', lost_items=lost_items, found_items=found_items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
