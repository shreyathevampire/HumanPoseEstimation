# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import imutils
from imutils.object_detection import non_max_suppression

'''The algorithm for this script is the same,
but it is pieced together in such a way to allow for
processing every frame of your webcam stream.'''

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
labeldict = {0 : 'adho mukha svanasana',
				1: 'camatkarasana',
				2: 'makara adho mukha svanasana',
				3: 'padmasana'}


def detect_and_predict_mask(frame, model):
    '''Parameters:
    frame = A frame from VideoStream
    faceNet = the model used to detect faces in the image
    maskNet = face mask classifier'''

	# grab the dimensions of the frame and then construct a blob
	# from it
    # print(frame.shape)

    # (h, w) = frame.shape[:2]
    # image = cv2.imread(frame)
    image = imutils.resize(frame, width=min(400, frame.shape[1]))
    # orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
    	cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    print(rects)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print(pick)

    #draw the final bounding boxes
    # (xA,yA,xB,yB) = (x,y,x+1)
    label = "No prediction"
    for (xA, yA, xB, yB) in pick:
    	cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    	(startX, startY) = (max(0, xA), max(0, yA))
    	(endX, endY) = (min(w - 1, xB), min(h - 1, yB))
    	face = image[yA:yB, xA:xB]
    	face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    	face = cv2.resize(face, (224, 224))
    	# cv2.imshow("Face",face)
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
        # print(val)
    return (pick,label)
















# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
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

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
currentframe = 0
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()

    frame = imutils.resize(frame, width=400)
    print(frame.shape)
    # name = './data/frame'+str(currentframe)+'.jpg'
    # cv2.imwrite(name, frame)
    # print(name)
    #
    # # detect faces in the frame and determine if they are wearing a
    # # face mask or not
    (pos, label) = detect_and_predict_mask(frame, model)
    # (xA,yA,xB,yB) = pos
    # cv2.putText(frame, label, (xA, yA - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 2)
    #
    # # show the output images
    # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 255),2)

        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    currentframe += 1
    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
