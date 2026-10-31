from flask_sqlalchemy import SQLAlchemy

# Đảm bảo không bị xung đột import database
try:
    from database import db
except Exception:
    db = SQLAlchemy()

class LostItem(db.Model):
    __tablename__ = 'lost_item'

    id = db.Column(db.Integer, primary_key=True)
    ten_do = db.Column(db.String(100))
    loai_do = db.Column(db.String(100))
    noi_mat = db.Column(db.String(100))
    ngay_mat = db.Column(db.String(20))
    mau_sac = db.Column(db.String(50))
    hinh_anh = db.Column(db.String(200))
    trang_thai = db.Column(db.String(30), default="DANG_TIM")
    thoi_diem_tra = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return self.ten_do or ''
