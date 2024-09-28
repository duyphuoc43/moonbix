import cv2
import numpy as np
import mss
import time
from ultralytics import YOLO
import pyautogui

# Kích thước màn hình muốn quay
monitor_width = 380
monitor_height = 690
top = 10
left = 0
# Khởi tạo đối tượng mss để quay màn hình
sct = mss.mss()

# Xác định vùng màn hình cần quay
monitor = {"top": top, "left": left, "width": monitor_width, "height": monitor_height}

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

    # annotated_frame = results[0].plot()  # Vẽ các hộp bounding box lên frame
    # cv2.imshow("Screen Capture with YOLOv8 Detection", annotated_frame)

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
            
            mocs = detections[2]
            
            phithuyens = detections[3]
            

            x1 = int(phithuyens[0]["center_x"])
            y1 = int(phithuyens[0]["y_box"] + phithuyens[0]["h_box"])
            confidence_phithuyen = phithuyens[0]["confidence"]
            x2 = int(mocs[0]["center_x"])
            y2 = int(mocs[0]["center_y"])
            confidence_moc = mocs[0]["confidence"]

            for phithuyen in phithuyens:
                if(phithuyen["confidence"] > confidence_phithuyen):
                    x1 = int(phithuyen["center_x"])
                    y1 = int(phithuyen["y_box"] + phithuyen["h_box"])

            for moc in mocs:
                if(moc["confidence"] > confidence_moc):
                    x2 = int(moc["center_x"])
                    y2 = int(moc["y_box"] + moc["h_box"])
                    
            size = 5
            
            for item in gold:
                stop = False
               
                for i in range(-size, size):
                    if (stop):
                        break
                    x3 = int(item["center_x"]) + i
                    for j in range(-size, size):
                        y3 = int(item["center_y"]) + j
                        if((y2 - y1)*(x3 - x1) - (y3 - y1)*(x2 - x1) == 0):
                            pyautogui.click(150, 350)
                            print(x1,y1,x2,y2,x3,y3)
                            print("click")
                            stop = True
                # time.sleep(1)
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng các tài nguyên
cv2.destroyAllWindows()
