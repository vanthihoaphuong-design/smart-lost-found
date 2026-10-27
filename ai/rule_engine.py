def improve_result(yolo_name, info):

    color = info.get("color", "")
    shape = info.get("shape", "")

    name = yolo_name.lower()

    # -----------------------------
    # YOLO không nhận được
    # -----------------------------
    if yolo_name == "Không xác định":

        if shape == "Tròn" and color in ["Trắng", "Xám"]:
            return "Bình giữ nhiệt"

        if shape == "Tròn" and color == "Xanh dương":
            return "Chai nước"

        return "Không xác định"

    # -----------------------------
    # bottle
    # -----------------------------
    if name == "bottle":

        if color == "Trắng":
            return "Bình giữ nhiệt"

        return "Chai nước"

    # -----------------------------
    # vase
    # -----------------------------
    if name == "vase":

        return "Chai nước"

    # -----------------------------
    # cup
    # -----------------------------
    if name == "cup":

        return "Ly"

    # -----------------------------
    # mouse
    # -----------------------------
    if name == "mouse":

        return "Chuột máy tính"

    # -----------------------------
    # backpack
    # -----------------------------
    if name == "backpack":

        return "Ba lô"

    # -----------------------------
    # handbag
    # -----------------------------
    if name == "handbag":

        return "Túi xách"

    # -----------------------------
    # suitcase
    # -----------------------------
    if name == "suitcase":

        return "Va li"

    # -----------------------------
    # laptop
    # -----------------------------
    if name == "laptop":

        return "Laptop"

    # -----------------------------
    # cell phone
    # -----------------------------
    if name == "cell phone":

        return "Điện thoại"

    return yolo_name