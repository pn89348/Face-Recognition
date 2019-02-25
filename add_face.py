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

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.75
white = (255, 255, 255)
cyan = (255, 255, 0)
red = (0, 0, 255)
stroke1 = 1
stroke2 = 2
stroke3 = 3

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Convert to grayscale (black and white)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Finding Faces
	faces = face_cascade.detectMultiScale(gray)

	num_faces = len(faces)

	# Give Message if No Faces are Detected
	if num_faces == 0:
		text = "No Faces Detected. Training Paused Temporarily."
		cv2.putText(frame, text, (0, 20), font, font_size, red, stroke2, cv2.LINE_AA)

	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+h]

		# Pause Data Collection if Multiple Faces are Detected
		if num_faces == 1:
			count += 1
			file = directory + "/" + str(count) + ".png"
			cv2.imwrite(file, roi_color)
		else:
			text = str(num_faces) + " Faces Detected. Training Paused Temporarily."
			cv2.putText(frame, text, (0, 20), font, font_size, red, stroke2, cv2.LINE_AA)

		# Display count
		text = str(count) + "/" + str(num_pics)
		cv2.putText(frame, text, (x, y), font, font_size, white, stroke2, cv2.LINE_AA)

		# Draw Rectangle around face
		x_end = x + w
		y_end = y + h
		cv2.rectangle(frame, (x, y), (x_end, y_end), cyan, stroke3)
	

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
