from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

translate = {

    # Người
    "person": "Người",

    # Đồ điện tử
    "cell phone": "Điện thoại",
    "laptop": "Laptop",
    "mouse": "Chuột máy tính",
    "keyboard": "Bàn phím",
    "remote": "Điều khiển",
    "tv": "Màn hình",
    "tvmonitor": "Màn hình",

    # Học tập
    "book": "Sách",
    "backpack": "Ba lô",
    "handbag": "Túi xách",
    "suitcase": "Vali",
    "umbrella": "Ô",
    "tie": "Cà vạt",

    # Đồ dùng
    "bottle": "Bình nước",
    "cup": "Ly",
    "bowl": "Bát",
    "vase": "Bình hoa",
    "clock": "Đồng hồ",
    "scissors": "Kéo",
    "teddy bear": "Gấu bông",

    # Nội thất
    "chair": "Ghế",
    "couch": "Ghế sofa",
    "bed": "Giường",
    "dining table": "Bàn",
    "toilet": "Bồn cầu",

    # Thiết bị
    "microwave": "Lò vi sóng",
    "oven": "Lò nướng",
    "toaster": "Máy nướng bánh",
    "sink": "Bồn rửa",
    "refrigerator": "Tủ lạnh",

    # Thể thao
    "sports ball": "Bóng",
    "baseball bat": "Gậy bóng chày",
    "baseball glove": "Găng tay",
    "tennis racket": "Vợt tennis",
    "frisbee": "Đĩa bay",
    "kite": "Diều",

    # Phương tiện
    "bicycle": "Xe đạp",
    "motorcycle": "Xe máy",
    "car": "Ô tô",
    "bus": "Xe buýt",
    "truck": "Xe tải",

    # Động vật
    "cat": "Mèo",
    "dog": "Chó",
    "bird": "Chim",

    # Thực phẩm
    "banana": "Chuối",
    "apple": "Táo",
    "orange": "Cam",
    "sandwich": "Bánh mì",
    "pizza": "Pizza",
    "cake": "Bánh ngọt",
        "donut": "Bánh donut",
    "hot dog": "Hotdog",

    # Quần áo
    "shirt": "Áo",
    "t-shirt": "Áo thun",
    "pants": "Quần",
    "shoe": "Giày",
    "sneaker": "Giày thể thao",
    "hat": "Mũ",

    # Khác
    "bench": "Ghế đá",
    "bag": "Túi",
    "wallet": "Ví",
    "key": "Chìa khóa"
}


def detect_object(image_path):
    """
    AI nhận diện đồ vật.

    Trả về:
        (ten_do, do_tin_cay)

    Ví dụ:
        ("Chuột máy tính", 97.35)
    """

    try:

        if not os.path.exists(image_path):
            return "Không xác định", 0

        results = model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            imgsz=640,
            verbose=False
        )

        if not results:
            return "Không xác định", 0

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return "Không xác định", 0

        # Lấy vật thể có độ tin cậy cao nhất
        best_box = max(
            result.boxes,
            key=lambda b: float(b.conf[0])
        )

        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])

        english_name = model.names[cls_id]

        vietnamese_name = translate.get(
            english_name,
            english_name
        )

        confidence = round(conf * 100, 2)

        print("=" * 40)
        print("AI DETECT")
        print("English :", english_name)
        print("Vietnam :", vietnamese_name)
        print("Score   :", confidence)
        print("=" * 40)

        return vietnamese_name, confidence

    except Exception as e:

        print("========== AI ERROR ==========")
        print(e)
        print("==============================")

        return "Không xác định", 0