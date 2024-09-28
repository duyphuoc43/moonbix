import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO
import pyautogui

# Kích thước màn hình muốn quay
monitor_width = 370
monitor_height = 700

# Khởi tạo đối tượng mss để quay màn hình
sct = mss.mss()

# Xác định vùng màn hình cần quay
monitor = {"top": 0, "left": 0, "width": monitor_width, "height": monitor_height}

# Load mô hình YOLOv8
model = YOLO("model/best1.pt")
while True:
    # Bắt đầu đo thời gian để tính FPS (tùy chọn)
    start_time = time.time()
    
    # Chụp một frame từ màn hình
    screenshot = sct.grab(monitor)

    # Chuyển đổi từ raw pixels sang định dạng numpy array
    img = np.array(screenshot)

    # Chuyển đổi từ BGRA (vì mss trả về BGRA) sang BGR cho OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Sử dụng mô hình YOLO để phát hiện đối tượng
    results = model(img, verbose=False)

    annotated_frame = results[0].plot()  # Vẽ các hộp bounding box lên frame
    cv2.imshow("Screen Capture with YOLOv8 Detection", annotated_frame)

    for result in results:
        detections = [[],[],[],[]]
        if result.boxes is not None:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = [coord.item() for coord in box.xyxy[0]]
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                detection = {
                    'x_box' : x_min,
                    'y_box' : y_min,
                    'w_box' : x_max-x_min,
                    'h_box' : y_max-y_min,
                    'center_x' : x_min + (x_max-x_min)/2,
                    'center_y' : y_min + (y_max-y_min)/2,
                    'confidence': round(confidence, 2),
                    'class_id': class_id,
                }
                detections[detection['class_id']].append(detection)

        boom = detections[0]
        gold = detections[1]

        if(len(detections[2]) > 0 and len(detections[3]) > 0):
            moc = detections[2]
            phithuyen = detections[3]

            x1 = int(phithuyen[0]["center_x"])
            y1 = int(phithuyen[0]["y_box"] + phithuyen[0]["h_box"])
            x2 = int(moc[0]["center_x"])
            y2 = int(moc[0]["center_y"])
            print(x1,y1,x2,y2)
            for item in gold:
                for i in range(-10, 10):
                    x3 = int(item["center_x"]) + i
                    for j in range(-10, 10):
                        y3 = int(item["center_y"]) + j
                        if((y2 - y1)*(x3 - x1) - (y3 - y1)*(x2 - x1) == 0):
                            pyautogui.click(150, 350)
                            print("click")
                            time.sleep(2)

    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
cv2.destroyAllWindows()
