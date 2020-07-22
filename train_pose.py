#importing the necessary package
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
'''imutils.paths will help us to find and list images in our dataset'''
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd

'''parse a few command line arguments to launch the script from terminal'''
ag = argparse.ArgumentParser()
ag.add_argument("-d","--dataset", required = True, help = "path to input dataset")
ag.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ag.add_argument("-m", "--model", type=str, default="pose_detector.h5", help="path to output pose detector model")
args = vars(ag.parse_args())


'''initialize learningrate,
number of epochs,
batchsize'''
INIT_LR = 1e-4
BS = 32
EPOCHS = 200


'''load the images'''
print("load images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels= []


'''preprocessing steps:
1. resizing the image to 224x224
conversion to array format
scaling the intensities in the input image to the range of [-1,1] => done using preprocess_input function'''

for imagePath in imagePaths:
    print(imagePath)
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size = (224,224))
    image  = img_to_array(image)
    image  = preprocess_input(image)

    data.append(image.tolist())
    print(type(image))
    labels.append(label)
    print(label)

data = np.array(data, dtype = "float32")
labels = np.array(labels).reshape(-1,1)
print(" ==============labels =========")
print(labels.shape)
print(data.shape)

'''encode labels
split dataset
data augmentation'''
#one hot encoding
lb  = MultiLabelBinarizer()
label = lb.fit_transform(labels)
print(labels.shape)
print(label)
# label = to_categorical(label,num_classes=4)
print(label)




'''split dataset'''
(trainX, testX, trainY, testY) = train_test_split(data, label,
	test_size=0.20, random_state=42)

print("[INFO] trainX images shape = ",trainX.shape)
print("[INFO] trainY images shape = ",trainY.shape)
print("[INFO] testY images shape = ",testY.shape)
print("[INFO] testX images shape = ",testX.shape)


''' construct the training image generator for data augmentation'''
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# total = 0
# toal_img = 10
# print("[INFO] generating images...")
# os.mkdir("genimages")
#
# image = load_img('/home/ubuntu/Desktop/desktop/summerintern/yoga-poses-dataset/temp_dataset/padmasana/lotus.png')
# image = img_to_array(image)
# image = np.expand_dims(image, axis =0)
# imageGen = aug.flow(image,save_to_dir = 'genimages', save_prefix = "aug", save_format = "png")

# for image in imageGen:
#     total+= 1
#
#     if total == toal_img:
#         break
# aug.fit(trainX)

'''during training we will apply data augmentation
ie. on the fly dara mutations to improve geeralizations'''


'''import mobilenet_v2 for fine tuning'''
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(8, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)


# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False


 # compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
print("[INFO] performing on-the-fly data augmentation")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# serialize the model to disk
print("[INFO] saving pose detector model...")
# serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
model.save(args["model"], save_format="h5")


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
