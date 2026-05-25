import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from model import ZeroShotPatternDetector

# Khởi tạo detector
detector = ZeroShotPatternDetector()

def inference_pipeline(pattern_img, drawing_img, threshold):
    if pattern_img is None or drawing_img is None:
        return None, {"Thông báo": "Vui lòng upload đầy đủ ảnh mẫu và bản vẽ!"}
        
    try:
        # Tạo file tạm an toàn bằng tempfile để tránh xung đột đa luồng
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_p, \
             tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_d:
            
            temp_p_path = tmp_p.name
            temp_d_path = tmp_d.name

        # Lưu ảnh từ mảng numpy array của Gradio thành file vật lý
        # Gradio mặc định trả về RGB -> Chuyển sang BGR cho OpenCV xử lý chính xác
        cv2.imwrite(temp_p_path, cv2.cvtColor(pattern_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(temp_d_path, cv2.cvtColor(drawing_img, cv2.COLOR_RGB2BGR))
        
        # === FIX CHÍNH Ở ĐÂY ===
        # Hứng đủ 3 giá trị trả về từ model mới: boxes, scores, labels
        boxes, scores, labels = detector.detect(temp_p_path, temp_d_path, conf_threshold=threshold)
        
        output_img = drawing_img.copy()
        results_json = []
        
        if len(boxes) > 0:
            # Duyệt qua cả nhãn (label) để vẽ lên màn hình
            for box, score, label in zip(boxes, scores, labels):
                x, y, w, h = box
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Thiết lập màu xanh lá sắc nét cho linh kiện
                color = (0, 255, 0)
                
                # Vẽ bounding box lên đối tượng tìm thấy
                cv2.rectangle(output_img, (x, y), (x + w, y + h), color, 2)
                
                # Hiển thị tên Linh kiện + điểm số (Ví dụ: FUSE: 0.85)
                text = f"{label}: {score:.2f}"
                cv2.putText(output_img, text, (x, max(15, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Định dạng chuẩn đầu ra cấu trúc JSON
                results_json.append({
                    "box_2d": [y, x, y + h, x + w],
                    "label": label,
                    "confidence": round(score, 2)
                })
        else:
            # Giải phóng dung lượng file tạm ngay cả khi không tìm thấy kết quả
            if os.path.exists(temp_p_path): os.remove(temp_p_path)
            if os.path.exists(temp_d_path): os.remove(temp_d_path)
            return output_img, {"message": "Không tìm thấy pattern nào phù hợp với ngưỡng hiện tại."}
            
        # Dọn dẹp file tạm sau khi chạy xong để tránh tràn bộ nhớ ổ cứng
        if os.path.exists(temp_p_path): os.remove(temp_p_path)
        if os.path.exists(temp_d_path): os.remove(temp_d_path)
            
        return output_img, results_json
        
    except Exception as e:
        return None, {"Lỗi hệ thống": str(e)}

# Khởi dựng giao diện Gradio UI Blocks
with gr.Blocks(title="Zero-Shot Pattern Detection in Drawings") as demo:
    gr.Markdown("# 🔍 Zero-Shot Pattern Detection trong Bản vẽ Kỹ thuật")
    gr.Markdown("Hệ thống tự động phát hiện các ký hiệu linh kiện kỹ thuật dựa trên ảnh mẫu (Template) mà không cần huấn luyện lại.")
    
    with gr.Row():
        with gr.Column():
            pattern_input = gr.Image(label="Upload Ký hiệu mẫu (Pattern / Query)", type="numpy")
            drawing_input = gr.Image(label="Upload Bản vẽ Kỹ thuật (Drawing)", type="numpy")
            threshold_slider = gr.Slider(minimum=0.2, maximum=0.9, value=0.60, step=0.05, label="Confidence Threshold")
            submit_btn = gr.Button("Chạy Trích xuất Linh kiện", variant="primary")
            
        with gr.Column():
            image_output = gr.Image(label="Kết quả Trực quan (Bounding Boxes)")
            json_output = gr.JSON(label="Tọa độ BBox & Scores (Cấu trúc JSON)")
            
    submit_btn.click(
        fn=inference_pipeline,
        inputs=[pattern_input, drawing_input, threshold_slider],
        outputs=[image_output, json_output]
    )

if __name__ == "__main__":
    demo.launch()