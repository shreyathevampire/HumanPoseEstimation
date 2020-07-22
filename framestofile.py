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




data = []
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

print("DONE")
