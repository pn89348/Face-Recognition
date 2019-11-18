# This program will take a specified number of pictures of a person and store it.

# Import Libraries
import cv2
import os
import time

name = input("Enter your name: ").title()
directory = name.replace(" ", "-")
directory = "../Images/" + directory

try:
	os.mkdir(directory)
except FileExistsError:
	print("The folder %s already exists! Overwriting." % directory)
else:
	print("Succesfully created folder %s!" % directory)

# Specify the Classifier for the Cascade
face_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_default.xml')
start_time = time.time()

# Start Capturing Video from Default Webcam
cap = cv2.VideoCapture(0)

count = 0
num_pics = 100

font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.75
white = (255, 255, 255)
cyan = (255, 255, 0)
red = (0, 0, 255)

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
		text1 = "No Face Detected!"
		text2 = "Please make sure " + name + " is in frame."
		cv2.putText(frame, text1, (0, 20), font, font_size, red, 2, cv2.LINE_AA)
		cv2.putText(frame, text2, (0, 40), font, font_size, red, 2, cv2.LINE_AA)

	for (x, y, w, h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+h]

		# Give Message and Pause Data Collection if Multiple Faces are Detected
		if num_faces == 1:
			count += 1
			file = directory + "/" + str(count) + ".png"
			cv2.imwrite(file, roi_color)
		else:
			text1 = str(num_faces) + " Faces Detected! Data Collection Paused."
			text2 = "Please make sure only " + name + " is in frame."
			cv2.putText(frame, text1, (0, 20), font, font_size, red, 2, cv2.LINE_AA)
			cv2.putText(frame, text2, (0, 40), font, font_size, red, 2, cv2.LINE_AA)


		# Display count
		text = str(count) + "/" + str(num_pics)
		cv2.putText(frame, text, (x, y), font, font_size, white, 2, cv2.LINE_AA)

		# Draw Rectangle around face
		x_end = x + w
		y_end = y + h
		cv2.rectangle(frame, (x, y), (x_end, y_end), cyan, 3)
	

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

end_time = time.time()
total_time = end_time - start_time
print("Program complete in %f seconds" % total_time)