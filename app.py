import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super-secret-key')

# 1. Cấu hình kết nối Cloud PostgreSQL
db_url = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 2. Bọc OpenCV/NumPy an toàn
try:
    import cv2
    import numpy as np
except Exception as e:
    print(f"Lỗi khởi tạo OpenCV/NumPy: {e}")

# 3. Import các Model Database
from models.lost_item import LostItem
from models.found_item import FoundItem

# 4. Tự động tạo bảng trên Database PostgreSQL
with app.app_context():
    try:
        db.create_all()
        print("Đã khởi tạo thành công các bảng database trên Cloud PostgreSQL!")
    except Exception as e:
        print(f"Lỗi tạo bảng database: {e}")

# ==========================================
# CÁC ROUTE ĐÃ ĐƯỢC KHỚP CHUẨN FILE HTML
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lost', methods=['GET', 'POST'])
def lost_item():
    if request.method == 'POST':
        pass
    return render_template('lost_form.html')  # Đã sửa thành lost_form.html

@app.route('/found', methods=['GET', 'POST'])
def found_item():
    if request.method == 'POST':
        pass
    return render_template('found_form.html')  # Đã sửa thành found_form.html

@app.route('/lost-list')
def lost_list():
    items = LostItem.query.order_by(LostItem.id.desc()).all()
    return render_template('list_items.html', items=items)  # Đã sửa thành list_items.html

@app.route('/found-list')
def found_list():
    items = FoundItem.query.order_by(FoundItem.id.desc()).all()
    return render_template('found_list.html', items=items)  # Đã sửa thành found_list.html

@app.route('/match')
def match_items():
    lost_items = LostItem.query.all()
    found_items = FoundItem.query.all()
    return render_template('match.html', lost_items=lost_items, found_items=found_items)

@app.route('/dashboard')
def dashboard():
    total_lost = LostItem.query.count()
    total_found = FoundItem.query.count()
    return render_template('dashboard.html', total_lost=total_lost, total_found=total_found)

@app.route('/heatmap')
def heatmap():
    lost_items = LostItem.query.all()
    found_items = FoundItem.query.all()
    return render_template('heatmap.html', lost_items=lost_items, found_items=found_items)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
