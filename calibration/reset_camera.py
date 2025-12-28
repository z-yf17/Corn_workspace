import time
import pyrealsense2 as rs

serial = "243722072992"

ctx = rs.context()
devs = ctx.query_devices()
found = False

for dev in devs:
    try:
        s = dev.get_info(rs.camera_info.serial_number)
        if s == serial:
            print("Resetting", serial)
            dev.hardware_reset()
            found = True
            break
    except Exception as e:
        print("err:", e)

if not found:
    print("Device not found:", serial)

time.sleep(5)
print("done")

