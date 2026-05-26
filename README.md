# 🔍 Zero-Shot Pattern Detection in BOM/CAD Drawings

Dự án này là một hệ thống Computer Vision ứng dụng kỹ thuật **Zero-Shot Learning** để tự động nhận diện, định vị và trích xuất tọa độ của bất kỳ ký hiệu kỹ thuật nào (điện trở, cầu chì, diode,...) trên bản vẽ mạch điện (BOM/CAD) dựa vào một ảnh mẫu (Template) đầu vào, **hoàn toàn không cần huấn luyện (training) lại hay tinh chỉnh (fine-tuning) mô hình**.

## 🌟 Đặc điểm nổi bật (Key Features)

* **Zero-Shot Detection:** Khám phá và định vị ký hiệu mới ngay lập tức chỉ với 1 ảnh mẫu (query image) duy nhất.
* **Precision Mode (Chống bẫy khoảng trắng):** Áp dụng kỹ thuật đảo ngược không gian màu (Invert Grayscale) kết hợp tạo mặt nạ (Masking). Hệ thống ép thuật toán chỉ tập trung vào các nét vẽ kỹ thuật màu đen, loại bỏ hoàn toàn nhiễu loạn từ phần nền trắng khổng lồ của bản vẽ.
* **Extreme Scale-Invariant (Bất biến tỷ lệ cực đại):** Thuật toán tự động nội suy và quét đa quy mô (Multi-scale) với 60 dải kích thước siêu mịn. Sử dụng phương pháp nội suy `cv2.INTER_AREA` để bảo toàn diện tích pixel, giúp nét mảnh không bị đứt gãy khi thu nhỏ.
* **Gaussian Tolerance (Dung sai hình học):** Xử lý triệt để sự chênh lệch độ dày nét vẽ giữa ảnh mẫu và bản vẽ thực tế bằng kỹ thuật làm mờ chủ động (Gaussian Blur), giúp Bounding Box bắt dính chính xác.
* **Giao diện trực quan:** Tích hợp Gradio UI cho phép người dùng tương tác trực tiếp và xuất kết quả tọa độ dưới định dạng JSON chuẩn.

## 📂 Cấu trúc thư mục

```text
zero-shot-pattern-detection/
│
├── examples/            # Thư mục chứa dữ liệu chạy thử nghiệm
│   ├── drawing1.png     # Các bản vẽ kỹ thuật nguyên bản
│   ├── drawing2.png
│   └── drawing3.png
│
├── src/                 # Mã nguồn chính của hệ thống
│   ├── app.py           # Giao diện Gradio và Pipeline inference
│   ├── crops.py         # Module hỗ trợ tiền xử lý/cắt ảnh tự động
│   └── model.py         # Core Algorithm: Tối ưu hóa Normalized Cross-Correlation
│
├── README.md            # Tài liệu kỹ thuật
└── requirements.txt     # Danh sách thư viện phụ thuộc
```

## 🧠 Quyết định Kỹ thuật (Technical Approach & Troubleshooting)

Trong quá trình phát triển, phương pháp **Deep Feature Extraction** (sử dụng Vision Transformer) đã được thử nghiệm. Tuy nhiên, qua quá trình phân tích miền dữ liệu (Domain Data Analysis) của bản vẽ CAD, các mô hình Deep Learning bộc lộ điểm yếu "Mù ngữ nghĩa vi mô" (Semantic Blindness): nét vẽ quá mảnh (1-2 pixels) bị chìm trong 95% diện tích khoảng trắng, gây ra hiện tượng nhận diện nhầm các khoảng trống thành biểu tượng.

Hệ thống đã được quyết định **Pivot (chuyển hướng)** sang sử dụng thuật toán **Pyramid Normalized Cross-Correlation (NCC)** kinh điển nhưng được tối ưu hóa ở mức độ sâu (Highly-optimized):

1. **Bóc tách kênh RGB:** Tự động loại bỏ chữ chú thích (thường có màu đỏ) và nền màu từ ảnh chụp màn hình người dùng tải lên, đưa ảnh mẫu về trạng thái tinh khiết nhất.
2. **Khắc phục "White-space Trap":** Lật ngược màu (Nền đen - Nét trắng) để `cv2.matchTemplate` không bị đánh lừa bởi các vùng không gian trống trên sơ đồ mạch.
3. **Hiệu năng (Performance):** Xử lý toàn bộ bản vẽ với đa tỷ lệ chỉ trong vòng chưa tới 1 giây trên CPU (vượt trội hoàn toàn về tốc độ và chi phí tính toán so với các mô hình Deep Learning nặng nề).

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

**Bước 1: Cài đặt thư viện**
Hệ thống yêu cầu Python 3.8+ và các thư viện xử lý ảnh, giao diện cơ bản.
```bash
pip install -r requirements.txt
```

**Bước 2: Khởi động giao diện Web**
Di chuyển vào thư mục `src` và chạy file `app.py`.
```bash
cd src
python app.py
```
Hệ thống sẽ cung cấp một đường dẫn Local URL (ví dụ: `http://127.0.0.1:7860/`). Bạn mở đường dẫn này trên trình duyệt để sử dụng.

## 💡 Hướng dẫn Sử dụng (Usage)

1. **Upload Bản vẽ (Drawing):** Tải lên sơ đồ mạch điện cần quét (sử dụng các ảnh trong thư mục `examples/`).
2. **Upload Ảnh mẫu (Pattern):** Tải lên ảnh ký hiệu cần tìm. *Hệ thống đã được tích hợp bộ tiền xử lý tự động làm sạch chữ và nền màu, tuy nhiên người dùng nên cắt ảnh tương đối gọn gàng để tối ưu tốc độ quét.*
3. **Điều chỉnh Confidence Threshold:** Mặc định được thiết lập ở mức `0.6` đến `0.65`. Kéo thanh trượt xuống nếu hệ thống bỏ sót (False Negatives), kéo lên nếu hệ thống nhận diện nhầm rác (False Positives).
4. **Trích xuất:** Kết quả trả về gồm 1 ảnh trực quan (đã vẽ Bounding Box màu xanh/đỏ) và 1 tệp JSON chứa tọa độ `[ymin, xmin, ymax, xmax]` tương ứng của từng ký hiệu.