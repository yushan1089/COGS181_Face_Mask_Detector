# import stuff
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

#from keras.preprocessing.image import img_to_array
#from keras.models import load_model
#from keras.preprocessing import image
import tkinter.filedialog as tkFileDialog

import argparse
import imutils
import time

import cv2
import keras
import numpy as np
import os

# restore the pre-trained model for mask prediction
mask_model = keras.models.load_model('SELF_DATASET_face_mask.h5')

# color dict for the rectangle shape
color_dict = {0:(255,0,0), 1:(0,0,255)}

# create the haar cascade for face detection

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

prototxtPath = os.path.sep.join(["./face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join(["./face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# initialize camera capture
cap = cv2.VideoCapture(0)
time.sleep(2.0)



while(True):
	# capture frame-by-frame
	ret, frame = cap.read()
	

	# some operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	facecas = faceCascade.detectMultiScale(gray, 1.1, 4)
	#print(facecas)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (160, 160))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))




			model_prediction = mask_model.predict(face.reshape(-1,160,160,3)) == mask_model.predict(face.reshape(-1,160,160,3)).max()
			my_result = np.sum(mask_model.predict(face.reshape(-1,160,160,3)), axis = 0)
			#print (my_result)
			if len(facecas) == 0:
				display_string = 'With Mask'
				color = (36,255,12)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			else:
				display_string = 'Without Mask'
				color = (0,0,255)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.putText(frame, display_string, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

	# display the resulting frame
	cv2.imshow('Live Face Mask Detection', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()