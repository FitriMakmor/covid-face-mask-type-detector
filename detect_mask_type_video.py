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

# Imports the required packages for the program
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


def detect_and_predict_mask_type(frame, faceNet, maskNet):
    # Retrieves the input video frame's spatial dimensions
    # and then creates a blob from the video frame.
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    # Passes the blob through the face detector neural network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Initializes the list variables storing the faces, their locations in the video frame
    # and the list of predictions from the face mask type neural network model
    faces = []
    locs = []
    preds = []

    # Loops over the face detections
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
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Adds the face and its bounding boxes to the
            # lists initialized earlier
            faces.append(face)
            locs.append((startX, startY, endX, endY))

        # Checks if there is at least one face in the detection
        if len(faces) > 0:
            # Executes Batch Predictions on ALL the faces at the same time
            # for faster inferences compared to doing the predictions one-by-one
            # in the `for` loop above

            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # Returns the face locations and their corresponding
        # predictions in the form of a 2-tuple
        return (locs, preds)


# Specifies the required arguments when running the program as well as
# its default value when no argument is given
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_type_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Loads the face detector model which
# uses a Caffe-based deep learning face detector
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loads the Face Mask Type Detector (FMTD) model from disk
print("[INFO] loading face mask type detector model...")
maskNet = load_model(args["model"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loops over the the video frames from the real-time video stream
while True:

    # Grabs the video frame from the threaded video stream and resizes it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detects the faces in the frame and determine if they are wearing a
    # surgical mask or a respirator
    (locs, preds) = detect_and_predict_mask_type(frame, faceNet, maskNet)

    # Loops over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):

        # Unpacks the bounding box and the associated predictions
        (startX, startY, endX, endY) = box
        (respirator, surgical_mask) = pred

        # Based on the results of the FMTD
        # adds a bounding box containing the label for the mask type and the color,
        # where purple signifies a disposable respirator, and blue for surgical mask
        label = "Respirator" if respirator > surgical_mask else "Surgical Mask"
        color = (194, 27, 228) if label == "Respirator" else (250, 238, 68)

        # Includes the probability of the prediction in the label
        label = "{}: {:.2f}%".format(label, max(respirator, surgical_mask) * 100)

        # Displays the label and the bounding box rectangles on
        # the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Shows the output frame containing the bounding box and the labels
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Breaks the loop when the `q` key is pressed
    if key == ord("q"):
        break

# Performs cleanup before finally stopping the program
cv2.destroyAllWindows()
vs.stop()
