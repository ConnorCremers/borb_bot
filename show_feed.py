import time

import cv2
from picamera2 import Picamera2, Preview


# Set up camera, make still configuration max resolution
camera = Picamera2()
camera_cfg = camera.create_still_configuration(main={'size': (3280, 2464)}, lores={'size': (640, 480)}, display='lores')
camera.configure(camera_cfg)

# Start showing what the camera sees
camera.start_preview(Preview.QTGL)
camera.start()

# Wait for keyboard interrupt (sure there is a better way to do this thru QTGL)
try:
    while True:
        time.sleep(100)
except KeyboardInterrupt:
    pass


# validate that we can run opencv commands on an image we capture from camera
frame = camera.capture_array()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
