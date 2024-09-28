import cv2
import numpy as np
import mss
import time

# Kích thước màn hình muốn quay
monitor_width = 370  # chiều rộng của vùng quay
monitor_height = 700  # chiều cao của vùng quay

# Khởi tạo đối tượng mss để quay màn hình
sct = mss.mss()

# Xác định vùng màn hình cần quay
monitor = {"top": 10, "left": 0, "width": monitor_width, "height": monitor_height}

while True:
    # Bắt đầu đo thời gian để tính FPS (tùy chọn)
    start_time = time.time()
    
    # Chụp một frame từ màn hình
    screenshot = sct.grab(monitor)

    # Chuyển đổi từ raw pixels sang định dạng numpy array
    img = np.array(screenshot)

    # Chuyển đổi từ BGRA (vì mss trả về BGRA) sang BGR cho OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Hiển thị frame
    cv2.imshow("Screen Capture", img)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # In FPS (tùy chọn)
    print("FPS: {}".format(1 / (time.time() - start_time)))

# Giải phóng các tài nguyên
cv2.destroyAllWindows()
