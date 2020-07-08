#import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import imutils
from imutils.object_detection import non_max_suppression




labeldict = {0 : 'adho mukha svanasana',
				1: 'camatkarasana',
				2: 'makara adho mukha svanasana',
				3: 'padmasana'}


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def send(imagePath, model, confidenceScore=50):

	image = cv2.imread(imagePath)
	print("[DEBUG] : image path = ",imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
	cv2.imshow("original input ",image)
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imshow("After NMS", image)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	print(rects)
	'''
	the gist of the non-maxima suppression algorithm is to take multiple, overlapping bounding boxes and reduce them to only a single bounding boxself.
	This helps reduce the number of false-positives reported by the final object detector.
	'''
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	print(pick)

	#draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		(startX, startY) = (max(0, xA), max(0, yA))
		(endX, endY) = (min(w - 1, xB), min(h - 1, yB))
		face = image[yA:yA+yB, xA:xA+xB]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		cv2.imshow("Face",face)
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		val = model.predict(face)
		print("prediction ")
		ind = np.argmax(val)
		posename = labeldict[ind];
		similarity_score = np.max(val)
		print("similarity_score = ",similarity_score)
		label = "{} : {:.2f}%".format(posename, similarity_score*100)
		print("index = ",ind," and corresponding label = ",labeldict[ind]);


		print(val)
		cv2.putText(image, label, (xA, yA - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)

		# show the output images
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 255),2)
		# cv2.imshow("Before NMS", orig)
		cv2.imshow("After NMS", image)
		cv2.waitKey()
		cv2.destroyAllWindows()
		return label


def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	ap.add_argument("-p", "--pose", type=str,
		default="pose_detector",
		help="path to pose detector model directory")
	ap.add_argument("-m", "--model", type=str,
		default="pose_detector.model",
		help="path to trained pose mask detector model")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	# load our serialized face detector model from disk
	print("[INFO] loading pose detector model...")
	prototxtPath = os.path.sep.join([args["pose"], "MobileNetSSD_deploy.prototxt"])
	print(prototxtPath)
	weightsPath = os.path.sep.join([args["pose"],
		"MobileNetSSD_deploy.caffemodel"])
	print(weightsPath)
	net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading pose mask detector model...")
	model = load_model(args["model"])
	imagePath = args["image"]


	label  = send(imagePath, model, args["confidence"])


if __name__ == '__main__':
	main()
