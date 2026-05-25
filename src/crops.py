import cv2

# Đọc tấm ảnh lớn chứa cả 3 phần
img = cv2.imread('/home/tuyen/projects/zero-shot-pattern-detection/examples/example3.png')
if img is None:
    raise FileNotFoundError('Không thể đọc ảnh. Kiểm tra lại đường dẫn: /home/tuyen/projects/zero-shot-pattern-detection/examples/example2.png')

h, w, _ = img.shape

# Tọa độ ước lượng dựa trên tỷ lệ ảnh bạn vừa gửi
# Cắt phần Pattern (Bên trái)
pattern = img[:, :int(w * 0.16)]

# Cắt phần Drawing (Ở giữa)
drawing = img[:, int(w * 0.16):int(w * 0.57)]

# Lưu thành 2 file riêng biệt để đưa vào mô hình
cv2.imwrite('pattern_query.png', pattern)
cv2.imwrite('drawing_input.png', drawing)

print("Đã cắt và lưu ảnh thành công!")