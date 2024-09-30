import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2, Preview


def detect_motion_avg_diff(frame1, frame2):
    """
    Calculate average pixel difference between two frames.

    Args:
    frame1 (numpy.ndarray): First frame
    frame2 (numpy.ndarray): Second frame

    Returns:
    bool: average difference between pixels
    """

    # Calculate absolute difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Calculate mean absolute difference (average pixel difference)
    avg_diff = np.mean(diff)

    # Check if average difference exceeds threshold
    return avg_diff


def detect_motion_num_px(frame1, frame2, thresh):
    """
    Calculate the number if pixels which are different by some threshold.

    Args:
    frame1 (numpy.ndarray): First frame
    frame2 (numpy.ndarray): Second frame
    threshold (int): Average pixel difference threshold

    Returns:
    bool: for each pixel, true if difference between frames is above threshold
    """

    # Calculate absolute difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Calculate mean absolute difference (average pixel difference)
    return np.sum(diff > thresh)


# Create a directory for saved images if it doesn't exist
image_dir = "captured_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize the Raspberry Pi camera, reduce buffer size for lower latency
camera = Picamera2()
camera_cfg = camera.create_still_configuration(
    main={'size': (820, 616)}, lores={'size': (640, 480)}, display='lores')
camera.configure(camera_cfg)

# Start showing what the camera sees
camera.start_preview(Preview.QTGL)
camera.start()

# Create list of N previously captured frames
prev_frame = camera.capture_array()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
N=15
prev_frames = [prev_frame] * N

# Initialization
start_time = time.time()
ii = 0
frame_count = 0

try:
    # Main loop
    while True:
        # Capture frame-by-frame
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save frame as JPEG image once per second
        current_time = time.time()
        # Calculate difference for all stored frames
        # We do it against a bunch of previous frames so hopefully periodic motion
        # (like feeder swinging) is not detected as a new frame
        diffs = [detect_motion_num_px(frame, prev_fr, 50) for prev_fr in prev_frames]
        if current_time - start_time >= 0.5 and min(diffs) > 3000:
            filename = f"image_{frame_count}.jpg"
            cv2.imwrite(os.path.join(image_dir, filename), frame)
            print(f"Saved image: {filename}")
            frame_count += 1
            start_time = current_time

        # Display the frame, update the appropriate previous frame
        prev_frames[ii] = frame
        ii += 1
        if ii == N:
            ii = 0

        time.sleep(0.1)

except KeyboardInterrupt:
    pass

# Release the camera and close any OpenCV windows
cv2.destroyAllWindows()
