import cv2
import numpy as np
import os

class ZeroShotPatternDetector:
    def __init__(self):
        print("🔧 ZeroShot Pattern Detector v2.2 - Fully Synchronized\n")

    def preprocess(self, img):
        """Nâng cao chất lượng ảnh, làm sắc nét đường vẽ mạch kỹ thuật"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        return sharpened

    def nms(self, boxes, scores, threshold=0.3):
        """Non-Maximum Suppression loại bỏ các khung trùng lặp đè lên nhau"""
        if len(boxes) == 0:
            return []

        boxes  = np.array(boxes,  dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
        areas  = (x2 - x1) * (y2 - y1)
        order  = scores.argsort()[::-1]
        keep   = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w   = np.maximum(0.0, xx2 - xx1)
            h   = np.maximum(0.0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)

            order = order[np.where(iou <= threshold)[0] + 1]

        return keep

    def detect(self, pattern_paths, drawing_path, conf_threshold=0.60):
        drawing = cv2.imread(drawing_path, cv2.IMREAD_GRAYSCALE)
        if drawing is None:
            raise FileNotFoundError(f"❌ Cannot read drawing: {drawing_path}")
        drawing = self.preprocess(drawing)
        orig_h, orig_w = drawing.shape

        all_boxes, all_scores, all_labels = [], [], []

        if isinstance(pattern_paths, str):
            pattern_paths = [pattern_paths]

        for pattern_path in pattern_paths:
            label   = os.path.basename(pattern_path).split('.')[0].upper()
            pattern = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
            if pattern is None:
                print(f"⚠️  Cannot read template: {pattern_path}")
                continue
            pattern = self.preprocess(pattern)

            # --- Cắt sát viền mẫu linh kiện ---
            _, thresh = cv2.threshold(pattern, 180, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                pattern = pattern[
                    max(0, y - 2): min(pattern.shape[0], y + h + 2),
                    max(0, x - 2): min(pattern.shape[1], x + w + 2),
                ]

            pat_h, pat_w = pattern.shape

            # Tự động tính toán dải tỉ lệ quét thích ứng kích thước bản vẽ
            min_target_w = 20
            max_target_w = min(orig_w // 3, 300)
            scale_min    = min_target_w / pat_w
            scale_max    = max_target_w / pat_w

            scales = np.linspace(scale_min, scale_max, 30)

            for scale in scales:
                sw = int(pat_w * scale)
                sh = int(pat_h * scale)
                if sw < 8 or sh < 8 or sw >= orig_w or sh >= orig_h:
                    continue

                resized = cv2.resize(pattern, (sw, sh), interpolation=cv2.INTER_AREA)
                res     = cv2.matchTemplate(drawing, resized, cv2.TM_CCOEFF_NORMED)
                loc     = np.where(res >= conf_threshold)

                rows, cols = loc
                for row, col in zip(rows, cols):
                    all_boxes.append([int(col), int(row), sw, sh])
                    all_scores.append(float(res[row, col]))
                    all_labels.append(label)

        if not all_boxes:
            return [], [], []

        keep = self.nms(all_boxes, all_scores, threshold=0.3)

        final_boxes  = [all_boxes[i]  for i in keep]
        final_scores = [all_scores[i] for i in keep]
        final_labels = [all_labels[i] for i in keep]

        return final_boxes, final_scores, final_labels