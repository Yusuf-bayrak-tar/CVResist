import cv2 as cv
import numpy as np

# HSV renk aralıkları ve isimler
Colour_Range = [
    [(0, 0, 0), (180, 80, 80), "BLACK", 0, (0, 0, 0)],
    [(10, 50, 20), (25, 200, 140), "BROWN", 1, (42, 42, 165)],
    [(0, 120, 100), (10, 255, 255), "RED", 2, (0, 0, 255)],
    [(10, 150, 100), (25, 255, 255), "ORANGE", 3, (0, 128, 255)],
    [(20, 80, 120), (40, 255, 255), "YELLOW", 4, (0, 255, 255)],
    [(35, 40, 40), (90, 255, 255), "GREEN", 5, (0, 255, 0)],
    [(90, 50, 50), (130, 255, 255), "BLUE", 6, (255, 0, 0)],
    [(130, 50, 50), (160, 255, 255), "VIOLET", 7, (255, 0, 127)],
    [(0, 0, 60), (180, 30, 170), "GRAY", 8, (128, 128, 128)],
    [(15, 30, 90), (30, 180, 200), "GOLD", -1, (0, 215, 255)],
    [(0, 0, 160), (180, 30, 220), "SILVER", -2, (192, 192, 192)],
]

Red_top_low = (160, 120, 100)
Red_top_high = (180, 255, 255)

def preprocess_image(img):
    img = cv.convertScaleAbs(img, alpha=1.4, beta=15)
    img = cv.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv.filter2D(img, -1, sharpen_kernel)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img = cv.cvtColor(cv.merge((cl,a,b)), cv.COLOR_LAB2BGR)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    s = np.clip(s.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
    img = cv.cvtColor(cv.merge([h, s, v]), cv.COLOR_HSV2BGR)
    img = cv.GaussianBlur(img, (3,3), 0)
    return img

def group_and_filter_bands(bands, dist_thresh=20):
    bands = sorted(bands, key=lambda b: b[0])
    grouped = []
    current = []

    for band in bands:
        if not current:
            current.append(band)
            continue
        if abs(band[0] - current[-1][0]) < dist_thresh:
            current.append(band)
        else:
            grouped.append(current)
            current = [band]
    if current:
        grouped.append(current)

    return [max(g, key=lambda b: b[4][2]*b[4][3]) for g in grouped]

def detect_bands(image):
    img = preprocess_image(image)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    height = img.shape[0]
    roi_top, roi_bot = int(height * 0.15), int(height * 0.85)
    raw_bands = []

    for low, high, name, val, color in Colour_Range:
        mask = cv.inRange(hsv, low, high)
        if name == "RED":
            mask |= cv.inRange(hsv, Red_top_low, Red_top_high)

        mask[:roi_top, :] = 0
        mask[roi_bot:, :] = 0
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h
            ratio = h / float(w + 1e-5)
            if 5 < w < 70 and 10 < h < height * 0.95 and area > 250 and 1.2 < ratio < 15:
                cx = x + w // 2
                raw_bands.append((cx, name, val, color, (x, y, w, h)))

    filtered = group_and_filter_bands(raw_bands)
    filtered = sorted(filtered, key=lambda b: b[0])[:6]
    return filtered