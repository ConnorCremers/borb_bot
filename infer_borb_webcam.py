import time
import os

import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from tflite_runtime.interpreter import Interpreter 

use_webcam = False

# Load up model
model_path = "model/detect.tflite"
interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, in_height, in_width, _ = input_details[0]['shape']
print("Image Shape (", in_width, ",", in_height, ")")

# Create a directory for saved images if it doesn't exist
image_dir = "captured_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize the Raspberry Pi camera
if use_webcam:
	camera = cv2.VideoCapture(0) # For using webcam
	camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
else:
	camera = Picamera2()
	camera_cfg = camera.create_still_configuration(
		main={'size': (3280, 2464)}, lores={'size': (640, 480)}, display='lores')
	camera.configure(camera_cfg)

# Start up camera & verify functionality
if use_webcam:
	if not camera.isOpened():
	print("Cannot open camera")
	exit()
else:
	camera.start()

with open('slack_token.txt') as file:
	slack_token = file.readline()
	slack_channel = file.readline()
slack_client = WebClient(token=slack_token)

frame_count = 0
start_time = 0

while True:
	current_time = time.time()
	# Capture frame-by-frame
	if use_webcam:
		_, frame = camera.read()
	else:
		frame = camera.capture_array()

	# Process image for feeding to NN
	image_for_det = cv2.resize(frame, (width, height))
	image_for_det = cv2.cvtColor(image_for_det, cv2.COLOR_BGR2RGB)
	input_data = np.expand_dims(image_for_det, axis=0)
	input_data = (np.float32(input_data) - 127.5) / 127.5

	# Run NN
	interpreter.set_tensor(input_details[0]['index'],input_data)
	interpreter.invoke()
	boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
	scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

	# Check if anything meets threshold
	for i, score in enumerate(scores):
		if score < .99:
			continue
		print(score)

		# Save out images
		filename = f"image_{frame_count}.jpg"
		frame_count += 1
		cv2.imwrite(os.path.join(image_dir, filename), frame)
		
		# Draw out box
		img_w = frame.shape[1]
		img_h = frame.shape[0]
		ymin = int(max(1,(boxes[i][0] * img_h)))
		xmin = int(max(1,(boxes[i][1] * img_w)))
		ymax = int(min(img_h,(boxes[i][2] * img_h)))
		xmax = int(min(img_w,(boxes[i][3] * img_w)))
		cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

		# If it has been 60 seconds since the last detection
		# send it to the specified slack channel
		if current_time - start_time >= 60:
			cv2.imwrite('bird_with_box.jpg', frame)
			res = slack_client.files_upload_v2(
				channel=slack_channel, 
				initial_comment=f'Borb Detected with score={score}!',
				file = 'bird_with_box.jpg'
			)
			start_time = current_time

	# Display the frame
	cv2.imshow('frame', frame)

	if cv2.waitKey(250) & 0xFF == ord('q'):
		break

# Cleanup
os.remove('bird_with_box.jpg')
cv2.destroyAllWindows()
