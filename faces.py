# This is the main program. It identifies the faces live from the webcam.

# Import Libraries
import cv2
import numpy as np
import pickle
from PIL import Image
import platform
if platform.system() == "Windows" or platform.system() == "Darwin":
	from PIL import ImageGrab # WINDOWS/MAC
else:
	import pyscreenshot as ImageGrab # LINUX

width, height = (1600, 900)

def input_method():
	face_input = input("Record from screen or webcam: ").lower()
	if "screen" in face_input or "display" in face_input:
		return "screen"
	elif "cam" in face_input:
		return "webcam"
	else:
		return input_method()

# Function to process frame
def process_frame(frame):
	# Convert to grayscale (black and white)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Finding Faces
	faces = face_cascade.detectMultiScale(gray)
	for (x, y, w, h) in faces:
		# print(x, y, w, h)

		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+h]

		# Convert to picture, Resize, Convert back to array
		img = Image.fromarray(roi_gray, 'L').resize((100, 100), Image.ANTIALIAS)
		roi_gray = np.array(img, "uint8")

		# Recognize
		id_, loss = recognizer.predict(roi_gray)
		if loss <= 95:
			print(labels[id_], "\t", loss)

			font = cv2.FONT_HERSHEY_SIMPLEX
			font_size = 0.75
			text = labels[id_]
			color = (0, 0, 255)
			stroke = 2
			cv2.putText(frame, text, (x, y), font, font_size, color, stroke, cv2.LINE_AA)
			cv2.putText(frame, str(round(loss, 3)), (x, int(y+h+20)), font, font_size, color, stroke, cv2.LINE_AA)

		# # Save picture of cropped face in a file called "my-image.png"
		# img_item = "my-image.png"
		# cv2.imwrite(img_item, roi_color)

		# Draw Rectangle around face
		color = (255, 255, 0) # BGR -> cyan
		stroke = 3
		x_end = x + w
		y_end = y + h
		cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
	return frame

input = input_method()

# Specify the Classifier for the Cascade
face_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_default.xml')

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("../trainer.yml")

labels = {}
with open("../labels.pickle", 'rb') as f:
	# Load File
	labels = pickle.load(f)

	# Invert dictionary (switch keys with values)
	labels = {value:key for key, value in labels.items()}

# Start Capturing Video from Default Webcam
webcam_capture = cv2.VideoCapture(0)

while True:
	## Capture frame-by-frame
	if input == "screen":
		screen_pil = ImageGrab.grab(bbox=(0, 0, 1600, 900))
		screen_np = np.array(screen_pil)
		frame = cv2.resize(cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB), (int(width/1.5), int(height/1.5)))
	else:
		ret, frame = webcam_capture.read()

	processed_frame = process_frame(frame)
	#processed_screen = cv2.resize(processed_screen_full, dsize=(800, 450), interpolation=cv2.INTER_LINEAR)

	# Display the resulting frame
	cv2.imshow(input.title(), processed_frame)

	# Stop if user presses 'q'
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# Stop getting webcam input and close all windows at the end of the program
webcam_capture.release()
cv2.destroyAllWindows()
