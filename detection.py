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
monitor = {"top": 10, "left": 0, "width": monitor_width, "height": monitor_height}

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
        # Hiển thị kết quả với hộp giới hạn
    annotated_frame = results[0].plot()  # Vẽ các hộp bounding box lên frame
    
    # Hiển thị frame
    cv2.imshow("Screen Capture with YOLOv8 Detection", annotated_frame)
    for result in results:
        print(len(results))
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
        if(len(detections[2]) > 0):
            moc = detections[2]
        else:
            moc = [{'x_box': 266.13214111328125,
            'y_box': 204.82144165039062,
            'w_box': 25.680084228515625,
            'h_box': 20.217391967773438,
            'center_x': 278.97218322753906,
            'center_y': 214.93013763427734,
            'confidence': 0.74,
            'class_id': 2}]
        if(len(detections[3]) > 0):
            phithuyen = detections[3]
        else:
            phithuyen = [{'x_box': 221.18624877929688,
                    'y_box': 144.13951110839844,
                    'w_box': 67.80099487304688,
                    'h_box': 55.251617431640625,
                    'center_x': 255.0867462158203,
                    'center_y': 171.76531982421875,
                    'confidence': 0.78,
                    'class_id': 3}]

        x1 = phithuyen[0]["center_x"]
        y1 = phithuyen[0]["center_x"]

        x2 = moc[0]["center_x"]
        y2 = moc[0]["center_x"]

        for item in gold:
            x3 = item["center_x"]
            y3 = item["center_x"]

            if((y2 - y1)*(x3 - x1) - (y3 - y1)*(x2 - x1) == 0):
                # print("true")
                pyautogui.click(150, 350)



    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # # In FPS (tùy chọn)
    # print("FPS: {}".format(1 / (time.time() - start_time)))

# Giải phóng các tài nguyên
cv2.destroyAllWindows()
