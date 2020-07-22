import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
# %matplotlib inline
import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from imutils import paths
from glob import glob
from tqdm import tqdm
import os
import argparse
from imutils import paths
import argparse
import pickle
import csv
from tf_pose import common
# import argparse
import logging
import sys
import time

# from tf_pose import common
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset csv file")
# ap.add_argument("-ml", "--tfmodel", required=True,
# 	help="path to output serialized model")
# ap.add_argument("-l", "--label-bin", required=True,
# 	help="path to output label binarizer")
ap.add_argument("-e", "--epochs", type=int, default=25,
	help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot1.png",
	help="path to output loss/accuracy plot")
ap.add_argument('--model', type=str, default='cmu',
                    help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
ap.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. '
                         'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
ap.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
args = ap.parse_args()

w, h = model_wh(args.resize)
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))



args1 = vars(ap.parse_args())





# videoPaths = list(paths.list_images(args1["dataset"]))
'''directoryPaths = os.listdir(args1["dataset"])
path = args1["dataset"]
print(directoryPaths)

for file in directoryPaths:
    print(file)
    label = file
    videoPaths = os.listdir(os.path.join(path,file))
    print(videoPaths)
    for i in tqdm(range(len(videoPaths))):
        count = 0
        videoFile = os.path.join(path,file,videoPaths[i])
        cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
        frameRate = cap.get(5) #frame rate
        print(frameRate)
        x=1
        print("[DEBUG] file path ",videoFile)
        while(cap.isOpened()):
            # print("CAP OPENED")
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                # storing the frames in a new folder named train_1
                filename ='train_1/' + videoFile.split('/')[-2].split(' ')[0] +"_frame%d.jpg" % count;count+=1
                cv2.imwrite(filename, frame)
        cap.release()

print("DONE")


# getting the names of all the images
# images = glob("train_1/*.jpg")
# train_image = []
# train_class = []'''


'''data = []
labels = []
body_dict = {'Nose': 0 , 'Neck': 2, 'RShoulder': 4, 'RElbow':6, 'RWrist':8, 'LShoulder':10 , 'LElbow':12 , 'LWrist':14, 'RHip':16, 'RKnee':18, 'RAnkle':20, 'LHip':22, 'LKnee':24, 'LAnkle':26, 'REye':28, 'LEye':30, 'REar':32, 'LEar': 34 }

imagePaths = os.listdir(args1["dataset"])
with open("yoga_poses_from_frames.csv","w") as fl:
    writer = csv.writer(fl)
    for i, filename in enumerate(imagePaths):
        try:
            print(filename)
            label = filename.split('_')[0]
            print(label)
            file = os.path.join(args1["dataset"],filename)
            print(file)
            # estimate human poses from a single image !
            image = common.read_imgfile(file, None, None)
            t = time.time()
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
            elapsed = time.time() - t

            logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

            body_arr = np.zeros(36)


            if humans:
                #for human in humans:
                print(humans[0].body_parts.values())
                for parts in humans[0].body_parts.values():
                    bp = str(parts.get_part_name()).split('.')[1]
                    x = parts.x
                    y = parts.y
                    print(bp,x,y)
                    body_arr[body_dict[bp]], body_arr[body_dict[bp]+1] = x,y
                print(body_arr)
                row = body_arr.tolist()
                row.append(label)
                writer.writerow(row)
        except AttributeError:
            print(str(file))

print("DONE")'''

#try predicting of a video
model = load_model('classification_model.h5')
labels = {0: 'BHUJ', 1: 'PADMASANA', 2: 'SHAVASANA', 3: 'TADASANA', 4: 'TRIKONASANA', 5: 'VRIKSH'}
body_dict = {'Nose': 0 , 'Neck': 2, 'RShoulder': 4, 'RElbow':6, 'RWrist':8, 'LShoulder':10 , 'LElbow':12 , 'LWrist':14, 'RHip':16, 'RKnee':18, 'RAnkle':20, 'LHip':22, 'LKnee':24, 'LAnkle':26, 'REye':28, 'LEye':30, 'REar':32, 'LEar': 34 }


def predictframe(filename):
	# files = glob("predict_1/*.jpg")
	# for filename in files:
	print(filename)
	image = common.read_imgfile(filename, None, None)
	t = time.time()
	humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
	elapsed = time.time() - t

	# logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

	body_arr = np.zeros(36)

	data = []
	if humans:
	    #for human in humans:
	    # print(humans[0].body_parts.values())
	    for parts in humans[0].body_parts.values():
	        bp = str(parts.get_part_name()).split('.')[1]
	        x = parts.x
	        y = parts.y
	        # print(bp,x,y)
	        body_arr[body_dict[bp]], body_arr[body_dict[bp]+1] = x,y
	    # print(body_arr)
	    row = body_arr.tolist()
	    data.append(row)
	    testX = np.array(data)
	    # print("[DEBUG] shape of testX ", testX.shape)
	    # testX = np.transpose(testX)
	    # print("[DEBUG] shape of testX after takin transpose", testX.shape)
	    testX = np.expand_dims(testX, axis=2)
	    # print("[DEBUG] shape of testX after expanding dimns", testX.shape)
	    print("[INFO] evaluating network...")
	    predictions = model.predict(x=testX.astype("float32"))
	    print(predictions)
	    val = np.max(predictions)*100
	    if val < 40:
	        print("NO ASANA")
	        return("NO ASANA")
	    else:
	        ind = np.argmax(predictions)
	        print("prediction for frame = ",labels[ind])
	        return(labels[ind])

	    # print(classification_report(testY.argmax(axis=1),
	    # 	predictions.argmax(axis=1), target_names=lb.classes_))
	    # row.append(label)

	else :
		print("Np human in the frame")
		return("NO HUMAN")
		# return("NO HUMAN")
    # if (frameId % math.floor(frameRate) == 0):
    #     # storing the frames in a new folder named train_1
    #     filename ='predict_1/' + videoFile.split('/')[-2].split(' ')[0] +"_frame%d.jpg" % count;count+=1

        # cv2.imwrite(filename, frame)




'''videoFile = os.path.join(args1["video"])
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
print(frameRate)
x=1
print("[DEBUG] file path ",videoFile)
count =0
writer = None
(W, H) = (None, None)
while(cap.isOpened()):
    # print("CAP OPENED")
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        print("No frames available")
        break
    if (frameId % math.floor(frameRate) == 0):
        # storing the frames in a new folder named train_1
        filename ='predict_1/' + videoFile.split('/')[-2].split(' ')[0] +"_frame%d.jpg" % count;count+=1

        cv2.imwrite(filename, frame)
        print("[DEBUG] frame name = ",filename)
        if W is None or H is None:
            (H,W) = frame.shape[:2]
        output = frame.copy()
        label = predictframe(filename)
        text = "activity: {}".format(label)
        cv2.putText(output,text, (35,50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,255,0),5)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer  = cv2.VideoWriter("output", fourcc, 30,
                (W,H), True)
        writer.write(output)
        cv2.imshow("OUTPUT", output)
print("INFO - cleaning up")
writer.release()
cap.release()'''



# initialize the video stream, pointer to output video file, and
# frame dimensions
videoFile = os.path.join(args1["video"])
vs = cv2.VideoCapture(args1["video"])
frameRate = vs.get(5) #frame rate
writer = None
(W, H) = (None, None)
count = 0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	frameId = vs.get(1) #current frame number
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if (frameId % math.floor(frameRate) == 0):
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		# clone the output frame, then convert it from BGR to RGB
		# ordering, resize the frame to a fixed 224x224, and then
		# perform mean subtraction
		output = frame.copy()
		filename ='predict_1/' + videoFile.split('/')[-2].split(' ')[0] +"_frame%d.jpg" % count;count+=1

		cv2.imwrite(filename, frame)

		preds = predictframe(filename)
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# frame = cv2.resize(frame, (224, 224)).astype("float32")
		# frame -= mean
		# draw the activity on the output frame
		text = "activity: {}".format(preds)
		cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
			1.25, (0, 255, 0), 5)
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter("outputvideo5.avi", fourcc, 30,
				(W, H), True)
		# write the output frame to disk
		writer.write(output)
		# show the output image
		# cv2.imshow("Output", output)
		# key = cv2.waitKey(1) & 0xFF
		# # if the `q` key was pressed, break from the loop
		# if key == ord("q"):
		# 	break
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
