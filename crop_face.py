# Import Libraries
import cv2
import os

directories = ['Images/Barack-Obama/', 'Images/Donald-Trump/', 'Images/Pranav-Nigam/']

# Making a function to crop the face
def crop_face(directory, image):
	# Specify the Classifier for the Cascade
	face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_default.xml')

	# Reading the given image
	img = cv2.imread(image)

	# Convert to grayscale (blace and white)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

	for (x, y, w, h) in faces:
		roi_color = img[y:y+h, x:x+h]

		img_file = image.split('/')[-1]
		img_item = directory + img_file
		cv2.imwrite(img_item, roi_color)

for directory in directories:
	images = os.listdir(directory)

	for image in images:
		file = directory + image
		crop_face(directory, file)