import pyautogui
import time

while(1):
    x, y = pyautogui.position()
    print(f"Vị trí chuột hiện tại: ({x}, {y})")

    # pyautogui.moveTo(500, 500, duration=1)

    # pyautogui.click()
    # pyautogui.click(x, y)