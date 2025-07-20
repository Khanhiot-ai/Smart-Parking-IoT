import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
# Tải model YOLOv8 đã được huấn luyện của bạn
model = YOLO('best.pt')

# Khởi tạo webcam USB (số 0 là camera đầu tiên tìm thấy)
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở được không
if not cap.isOpened():
    print("Lỗi: Không thể mở camera. Hãy kiểm tra lại kết nối.")
    exit()

print("Camera đã sẵn sàng. Nhấn 'q' trên cửa sổ video để thoát.")

# --- VÒNG LẶP CHÍNH ---
while True:
    # 1. Đọc một khung hình (frame) từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera. Kết thúc.")
        break

    # 2. Đưa khung hình vào model để nhận dạng
    # verbose=False để không in quá nhiều log ra màn hình
    results = model(frame, verbose=False)

    # 3. Xử lý kết quả trả về
    # results[0] chứa tất cả các thông tin nhận dạng trong khung hình này
    annotated_frame = results[0].plot()

    # 4. Hiển thị kết quả
    # Dòng results[0].plot() ở trên đã tự động vẽ các bounding box và nhãn
    # nên chúng ta chỉ cần hiển thị khung hình đã được chú thích đó
    cv2.imshow("Nhan dang bien so xe - YOLOv8", annotated_frame)

    # 5. Đợi người dùng nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp ---
# Giải phóng camera và đóng tất cả các cửa sổ
print("Đang đóng chương trình...")
cap.release()
cv2.destroyAllWindows()
