# This is the main program. It identifies the faces live from the webcam.

# Import Libraries
import cv2
import numpy as np
from PIL import Image
import pickle

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
cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

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

	# Display the resulting frame
	cv2.imshow('frame', frame)

	# Stop if user presses 'q'
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# Stop getting webcam input and close all windows at the end of the program
cap.release()
cv2.destroyAllWindows()
