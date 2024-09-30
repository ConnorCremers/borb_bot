import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter 


# Load up model
model_path = "model/detect.tflite"
interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
_, in_height, in_width, _ = input_details[0]['shape']
print("Image Shape (", in_width, ",", in_height, ")")

# Load an image to be classified.
image = cv2.imread("bird_1.jpg")

output_details = interpreter.get_output_details()

### Classify the image.
start_time = time.time()

#Modify image as needed to fit into network
image_mod = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_mod = cv2.resize(image_mod, (in_width, in_height))
input_data = np.expand_dims(image_mod, axis=0)
input_data = (np.float32(input_data) - 127.5) / 127.5 # using floating model, don't care about speed

# Set input data, run model
interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()
boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

end_time = time.time()

# Draw box around the detected bird(s)
for i, score in enumerate(scores):
	if score < .9:
		continue
	ymin = int(max(1,(boxes[i][0] * image.shape[0])))
	xmin = int(max(1,(boxes[i][1] * image.shape[1])))
	ymax = int(min(image.shape[0],(boxes[i][2] * image.shape[0])))
	xmax = int(min(image.shape[1],(boxes[i][3] * image.shape[1])))
	
	cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

# Display results
classification_time = np.round(end_time-start_time, 3)
print("Classificaiton Time =", classification_time, "seconds.")
cv2.imshow('Object detector', image)
while cv2.waitKey(0) != ord('q'):
	pass
