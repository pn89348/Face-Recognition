# This program will take a specified number of pictures of a person and store it.

# Import Libraries
import cv2
import os

directory = input("Enter your name: ").title().replace(" ", "-")
directory = "../Images/" + directory

try:
	os.mkdir(directory)
except FileExistsError:
	print("The folder %s already exists! Continuing program." % directory)
else:
	print("Succesfully created folder %s!" % directory)

# Specify the Classifier for the Cascade
face_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_default.xml')

# Start Capturing Video from Default Webcam
cap = cv2.VideoCapture(0)

count = 0
num_pics = 100

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Convert to grayscale (black and white)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Finding Faces
	faces = face_cascade.detectMultiScale(gray)
	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+h]

		count += 1
		file = directory + "/" + str(count) + ".png"
		cv2.imwrite(file, roi_color)

		# Display count
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_size = 0.75
		text = str(count) + "/" + str(num_pics)
		color = (255, 255, 255)
		stroke = 2
		cv2.putText(frame, text, (x, y), font, font_size, color, stroke, cv2.LINE_AA)

		# Draw Rectangle around face
		color = (255, 255, 0) # BGR -> cyan
		stroke = 3
		x_end = x + w
		y_end = y + h
		cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)

	# Display the resulting frame
	cv2.imshow('frame', frame)

	# Stop if user presses 'q' or specified numebr of pictures have been taken
	if (cv2.waitKey(20) & 0xFF == ord('q')) or count >= num_pics:
		break

# Stop getting webcam input and close all windows at the end of the program
cap.release()
cv2.destroyAllWindows()

if count >= num_pics:
	print("Face Data Collection Complete")
else:
	print("Canceled")
