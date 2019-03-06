# This program will train on each of the people that have data in the Images folder.

# Import Libraries
import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time

start_time = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # the program's directory
image_dir = os.path.join(BASE_DIR, "../Images") # the Images folder

face_cascade = cv2.CascadeClassifier('./Cascades/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(root)
			# print(label, path)

			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			# print(label_ids)
			
			# y_labels.append(label)
			# x_train.append(path)
			
			pil_image = Image.open(path).convert("L") # grayscale
			size = (100, 100)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(pil_image, "uint8")
			# print(image_array)
			faces = face_cascade.detectMultiScale(image_array)

			for(x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)
# print(y_labels)
# print(x_train)

with open("../labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("../trainer.yml")

end_time = time.time()
total_time = end_time - start_time
print("Successfully Trained in %f seconds" % total_time)