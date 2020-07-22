# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import csv
import pandas as pd
# import argparse
import logging
import sys
import time

# from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense

# logger = logging.getLogger('TfPoseEstimatorRun')
# logger.handlers.clear()
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True,
# 	help="path to input dataset csv file")
ap.add_argument("-m", "--tfmodel", required=True,
	help="path to output serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
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
# initialize the set of labels from the spots activity dataset we are
# going to train our network on
'''LABELS = set(["cobra pose", "crow pose", "dolphin plank pose", "downward facing dog pose", "eagle pose",
                "flip dog pose", "lotus pose", "low lunge pose"])
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args1["dataset"]))
data = []
labels = []
body_dict = {'Nose': 0 , 'Neck': 2, 'RShoulder': 4, 'RElbow':6, 'RWrist':8, 'LShoulder':10 , 'LElbow':12 , 'LWrist':14, 'RHip':16, 'RKnee':18, 'RAnkle':20, 'LHip':22, 'LKnee':24, 'LAnkle':26, 'REye':28, 'LEye':30, 'REar':32, 'LEar': 34 }

with open("yoga_poses.csv","w") as fl:
    writer = csv.writer(fl)
    for i, file in enumerate(imagePaths):
        try:
            print(file)
            label = file.split(os.path.sep)[-2]
            print(label)
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
            print(str(file))'''


data = pd.read_csv('yoga_poses_from_frames.csv')
labels = data.iloc[:,-1]
data = data.iloc[:,:-1]

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
data = np.array(data)
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)
# print(trainX.shape)
# trainX = trainX.reshape(1,-1)
# print(trainX.shape)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
# mean = np.array([123.68, 116.779, 103.939], dtype="float32")
# trainAug.mean = mean
# valAug.mean = mean

# trainX = trainX[1,trainX.shape[1],np.newaxis]
# print(trainX.shape)
# testX = testX[testX.shape[0],testX.shape[1],np.newaxis]
# print(testX.shape)

#create a model
model = Sequential()
model.add(Conv1D(32, 3, input_shape=(trainX.shape[1],1)))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(32, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
# model.add(LSTM(20))
# model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))



# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args1["epochs"])
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")

trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)
print("[DEBUG] checking shape of trainset input ",trainX.shape)
print("[DEBUG] checking shape of test set input ",testX.shape)
H = model.fit(x = trainX,
	y = trainY,
	batch_size = None,
	# x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	# validation_data=valAug.flow(testX, testY),
	validation_data = (testX,testY),
	validation_steps=len(testX) // 32,
	epochs=100)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
N = 100
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args1["plot"])

# serialize the model to disk
print("[INFO] serializing network...")
model.save(args1["tfmodel"])
# serialize the label binarizer to disk
f = open(args1["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
