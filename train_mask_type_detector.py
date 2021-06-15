# FACE MASK TYPE CLASSIFIER
# Author: Nur Muhammad Fitri Bin Makmor

# Description:
# Trains the Face Mask Type Detector (FMTD) model using the supplied dataset
# and plots a graph displaying its training accuracy and validation accuracy
# as well as its training loss and validation loss

# References:
#
# TITLE: COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning
# CONTRIBUTOR(S): Adrian Rosebrock (4th May 2020)
# LINK: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
#
# TITLE: Covid-19 Face Mask Detection Using TensorFlow, Keras and OpenCV
# CONTRIBUTOR(S): Arjya Das, Mohammad Wasif Ansari, and Rohini Basak (5th February 2021)
# LINK: https://ieeexplore.ieee.org/document/9342585

# ====================================================================================== #

# Imports the required packages for the program
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Specifies the required arguments when running the program as well as
# its default value when no argument is given
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_type_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# Initializes the initial learning rate, the number of epochs to train for,
# and the batch size
INIT_LR = 1e-4  # The initial learning rate
EPOCHS = 20  # Number of Epochs
BS = 32  # Batch Size

# Grabs the list of images in the dataset directory, and then initializes
# the list variables for the images (data) as well as its corresponding labels
# (Respirator or Surgical Mask)
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Loops over the image paths
for imagePath in imagePaths:

	# Extracts the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# Loads the input image at a 224x224 pixel size and then preprocesses it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# Updates the data and labels lists respectively
	data.append(image)
	labels.append(label)

# Converts the data and labels into NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Performs a one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Partitions the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Constructs the training image generator for the purpose of data augmentation
# where image is randomly rotated, zoomed, shifted, sheared, and horizontally flipped
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Loads the MobileNetV2 network, ensuring the head FC layer sets are
# left off from the model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Constructs the head of the model that will be placed on top of the
# the base model previously created
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Places the head FC model on top of the base model, becoming the actual
# model that will be trained
model = Model(inputs=baseModel.input, outputs=headModel)

# Loops over all the layers in the base model and perform fine tuning,
# where the base layers are frozen so that they will NOT be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compiles the FMTD model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Trains the head of the network using the specified batch size and epochs
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Makes predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Finds the index of the label with the corresponding largest predicted probability
# for each of the image in the testing set
predIdxs = np.argmax(predIdxs, axis=1)

# Shows a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Serializes the model into the disk
print("[INFO] saving mask type detector model...")
model.save(args["model"], save_format="h5")

# Plots the training loss and accuracy of the FMTD model using matplotlib
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
