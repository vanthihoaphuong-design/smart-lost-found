import cv2
import numpy as np


def detect_color(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    colors = {
        "Đỏ": ([0,120,70],[10,255,255]),
        "Cam": ([11,100,100],[25,255,255]),
        "Vàng": ([26,100,100],[34,255,255]),
        "Xanh lá": ([35,80,80],[85,255,255]),
        "Xanh dương": ([86,80,80],[125,255,255]),
        "Tím": ([126,80,80],[160,255,255]),
        "Trắng": ([0,0,180],[180,40,255]),
        "Đen": ([0,0,0],[180,255,50]),
        "Xám": ([0,0,50],[180,30,180])
    }

    best = "Không rõ"
    area_max = 0

    for name,(low,high) in colors.items():

        mask = cv2.inRange(
            hsv,
            np.array(low),
            np.array(high)
        )

        area = cv2.countNonZero(mask)

        if area > area_max:
            area_max = area
            best = name

    return best


def detect_shape(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,th = cv2.threshold(
        blur,
        120,
        255,
        cv2.THRESH_BINARY
    )

    contours,_ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours)==0:
        return "Không rõ"

    c=max(contours,key=cv2.contourArea)

    peri=cv2.arcLength(c,True)

    approx=cv2.approxPolyDP(c,0.04*peri,True)

    if len(approx)==3:
        return "Tam giác"

    elif len(approx)==4:
        return "Chữ nhật"

    elif len(approx)>6:
        return "Tròn"

    return "Khác"


def analyze(image_path):

    img=cv2.imread(image_path)

    if img is None:
        return {}

    return {

        "color":detect_color(img),

        "shape":detect_shape(img)

    }