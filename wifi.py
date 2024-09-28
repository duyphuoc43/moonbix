
import os

# Tắt Wi-Fi
def disable_wifi():
    os.system("networksetup -setairportpower airport off")

# Bật Wi-Fi
def enable_wifi():
    os.system("networksetup -setairportpower airport on")

# Tắt Wi-Fi
disable_wifi()

# Bật Wi-Fi
enable_wifi()
