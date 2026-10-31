from flask_sqlalchemy import SQLAlchemy

try:
    from database import db
except Exception:
    db = SQLAlchemy()

class FoundItem(db.Model):
    __tablename__ = 'found_item'

    id = db.Column(db.Integer, primary_key=True)
    ten_do = db.Column(db.String(100))
    loai_do = db.Column(db.String(100))
    noi_nhat = db.Column(db.String(200))
    ngay_nhat = db.Column(db.String(50))
    mau_sac = db.Column(db.String(50))
    hinh_anh = db.Column(db.String(300))
    trang_thai = db.Column(db.String(20), default="DANG_LUU_GIU")
    thoi_diem_tra = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return self.ten_do or ''
