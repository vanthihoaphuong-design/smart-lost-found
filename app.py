import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super-secret-key')

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
            # Lấy thông tin linh hoạt từ Form báo mất
            name = request.form.get('name') or request.form.get('item_name') or request.form.get('title')
            category = request.form.get('category') or request.form.get('type')
            description = request.form.get('description') or request.form.get('detail')
            location = request.form.get('location') or request.form.get('place')
            contact = request.form.get('contact') or request.form.get('phone')
            
            # Xử lý lưu ảnh
            file = request.files.get('image') or request.files.get('file')
            image_path = None
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_path = f"uploads/{filename}"

            # Khởi tạo đối tượng LostItem
            new_item = LostItem()
            if hasattr(new_item, 'name'): new_item.name = name
            if hasattr(new_item, 'title'): new_item.title = name
            if hasattr(new_item, 'category'): new_item.category = category
            if hasattr(new_item, 'description'): new_item.description = description
            if hasattr(new_item, 'location'): new_item.location = location
            if hasattr(new_item, 'contact'): new_item.contact = contact
            if hasattr(new_item, 'phone'): new_item.phone = contact
            if hasattr(new_item, 'image_path'): new_item.image_path = image_path
            if hasattr(new_item, 'image'): new_item.image = image_path

            db.session.add(new_item)
            db.session.commit()
            print(">>> LƯU THÀNH CÔNG BÀI BÁO MẤT ĐỒ!")
            return redirect(url_for('lost_list'))
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI KHI LƯU BÁO MẤT: {e}")
            
    return render_template('lost_form.html')

@app.route('/found', methods=['GET', 'POST'])
def found_item():
    if request.method == 'POST':
        try:
            # Lấy thông tin linh hoạt từ Form báo nhặt
            name = request.form.get('name') or request.form.get('item_name') or request.form.get('title')
            category = request.form.get('category') or request.form.get('type')
            description = request.form.get('description') or request.form.get('detail')
            location = request.form.get('location') or request.form.get('place')
            contact = request.form.get('contact') or request.form.get('phone')
            
            # Xử lý lưu ảnh
            file = request.files.get('image') or request.files.get('file')
            image_path = None
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                image_path = f"uploads/{filename}"

            # Khởi tạo đối tượng FoundItem
            new_item = FoundItem()
            if hasattr(new_item, 'name'): new_item.name = name
            if hasattr(new_item, 'title'): new_item.title = name
            if hasattr(new_item, 'category'): new_item.category = category
            if hasattr(new_item, 'description'): new_item.description = description
            if hasattr(new_item, 'location'): new_item.location = location
            if hasattr(new_item, 'contact'): new_item.contact = contact
            if hasattr(new_item, 'phone'): new_item.phone = contact
            if hasattr(new_item, 'image_path'): new_item.image_path = image_path
            if hasattr(new_item, 'image'): new_item.image = image_path

            db.session.add(new_item)
            db.session.commit()
            print(">>> LƯU THÀNH CÔNG BÀI BÁO NHẶT ĐỒ!")
            return redirect(url_for('found_list'))
        except Exception as e:
            db.session.rollback()
            print(f">>> LỖI KHI LƯU BÁO NHẶT: {e}")

    return render_template('found_form.html')

@app.route('/lost-list')
def lost_list():
    try:
        items = LostItem.query.order_by(LostItem.id.desc()).all()
    except Exception as e:
        items = []
        print(f"Lỗi lấy danh sách báo mất: {e}")
    return render_template('list_items.html', items=items, type='lost')

@app.route('/found-list')
def found_list():
    try:
        items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    except Exception as e:
        items = []
        print(f"Lỗi lấy danh sách báo nhặt: {e}")
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
