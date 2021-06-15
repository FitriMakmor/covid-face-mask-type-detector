# FACE MASK TYPE CLASSIFIER
# Author: Nur Muhammad Fitri Bin Makmor

# Description:
# Activates the configured default webcam and loop face mask type detection
# over the frames from the video stream.

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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

# Specify the required arguments when running the program as well as
# its default value when no argument is given
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_type_detector.model",
                help="path to trained face mask type detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loads the face detector model which
# uses a Caffe-based deep learning face detector
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loads the face mask type detector model from disk
print("[INFO] loading face mask type detector model...")
model = load_model(args["model"])

# Reads the image input, duplicates it and then retrieves its dimension sizes
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# Constructs a blob from the image input with sizes 300x300 and perform mean subtraction
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))

# Passes the blob through the face detector neural network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# Loops over the detections
for i in range(0, detections.shape[2]):
    # Extracts the confidence (i.e., probability) associated with
    # the detection
    confidence = detections[0, 0, i, 2]

    # Checks if the face detection is higher than the minimum confidence
    # specified in the argument to filter out weak detections
    if confidence > args["confidence"]:

        # Computes the bounding box values for the particular face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Ensures that the bounding boxes are within the
        # frame dimension size
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Extracts the face Regions Of Interest (ROI), converts it from BGR to RGB channel
        # ordering, resizes it to 224x224, and preprocesses it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Passes the pre-processed face to the face mask type detector model
        # to determine if it is wearing a respirator or a surgical mask
        (respirator, surgical_mask) = model.predict(face)[0]

        # Based on the results of the face mask type detector
        # adds a bounding box containing the label for the mask type and the color,
        # where purple signifies a disposable respirator, and blue for surgical mask
        label = "Respirator" if respirator > surgical_mask else "Surgical Mask"
        color = (194, 27, 228) if label == "Respirator" else (250, 238, 68)

        # Includes the probability of the prediction in the label
        label = "{}: {:.2f}%".format(label, max(respirator, surgical_mask) * 100)

        # Displays the label and the bounding box rectangles on
        # the output frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Optionally resizes the image
# image = cv2.resize(image, (600, 600))

# Optionally saves the image to a file path
cv2.imwrite("D:\Files\Documents\Courses\Sem 6\Soft Computing\Saved Output Images\output_image.jpg", image)

# Shows the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
