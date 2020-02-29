# A test program to test just the face recognizer code on a single image of a face.

# import the necessary packages
import numpy as np
import imutils
import pickle
import cv2
import os

# load our serialized face embedding model from disk
embedder = cv2.dnn.readNetFromTorch("recognizer/openface_nn4.small2.v1.t7")
#embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
#mbedder.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCV)
#embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
embedder.setPreferableTarget(cv2.dnn.DNN_BACKEND_OPENCV)

# load the face recognition model along with the label encoder
recognizer = pickle.loads(open("recognizer/recognizer.pickle", "rb").read())
le = pickle.loads(open("recognizer/le.pickle", "rb").read())

# load the image,
image = cv2.imread("jeffface.jpg")

# construct a blob for the face then pass the blob
# through our face embedding model 
faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (96, 96),
    (0, 0, 0), swapRB=True, crop=False)
embedder.setInput(faceBlob)

print ("about to run the recognition")
vec = embedder.forward()
print ("ran the recognition")

# perform classification to recognize the face
preds = recognizer.predict_proba(vec)[0]
j = np.argmax(preds)
proba = preds[j]
name = le.classes_[j]

# put the recognized name on the image
text = "{}: {:.2f}%".format(name, proba * 100)
cv2.putText(image, text, (10, 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)