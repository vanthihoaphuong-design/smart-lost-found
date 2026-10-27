import sqlite3
import os

# Database mà Flask thường sử dụng khi cấu hình:
# sqlite:///lost_found.db
DATABASE = os.path.join(
    os.path.dirname(__file__),
    "instance",
    "lost_found.db"
)

print("Database dang cap nhat:")
print(DATABASE)
print()

if not os.path.exists(DATABASE):
    print("❌ KHONG TIM THAY DATABASE!")
    print("Hay kiem tra thu muc instance.")
    input("Nhan Enter de thoat...")
    exit()

conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()

# =========================
# KIEM TRA BANG LOST_ITEM
# =========================

cursor.execute("""
    PRAGMA table_info(lost_item)
""")

columns = [row[1] for row in cursor.fetchall()]

print("Cac cot cua lost_item:")
print(columns)

if "thoi_diem_tra" not in columns:

    cursor.execute("""
        ALTER TABLE lost_item
        ADD COLUMN thoi_diem_tra DATETIME
    """)

    print("✅ DA THEM thoi_diem_tra vao lost_item")

else:

    print("ℹ️ lost_item DA CO thoi_diem_tra")


# =========================
# KIEM TRA BANG FOUND_ITEM
# =========================

cursor.execute("""
    PRAGMA table_info(found_item)
""")

columns = [row[1] for row in cursor.fetchall()]

print()
print("Cac cot cua found_item:")
print(columns)

if "thoi_diem_tra" not in columns:

    cursor.execute("""
        ALTER TABLE found_item
        ADD COLUMN thoi_diem_tra DATETIME
    """)

    print("✅ DA THEM thoi_diem_tra vao found_item")

else:

    print("ℹ️ found_item DA CO thoi_diem_tra")


# =========================
# LUU
# =========================

conn.commit()


# =========================
# KIEM TRA LAI
# =========================

print()
print("===== KIEM TRA SAU KHI CAP NHAT =====")

cursor.execute("""
    PRAGMA table_info(lost_item)
""")

print("lost_item:")
print([row[1] for row in cursor.fetchall()])


cursor.execute("""
    PRAGMA table_info(found_item)
""")

print("found_item:")
print([row[1] for row in cursor.fetchall()])


conn.close()

print()
print("====================================")
print("✅ CAP NHAT DATABASE THANH CONG")
print("✅ DU LIEU CU DUOC GIU NGUYEN")
print("====================================")