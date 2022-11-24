import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
# Setting an https context to fetch data from openml

if (not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Fetching the data

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
n_classes = len(classes)

# Splitting the data and scaling it

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

# Scaling the features

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

# Fitting the data into the model

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train_scaled, y_train)

# Calculating the accuracy of the model

y_pred = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("The accuracy is", accuracy)

# Starting the camera

cap = cv2.VideoCapture(0)
while (True):
    # Capture frame by frame

    try:
        ret, frame = cap.read()

        # Our operations on the frame come here

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Drawing a box in the center of the video

        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        # To only consider the area inside the box for detecting the digit

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # Converting cv2 image to pil format

        im_pil = Image.fromarray(roi)
        
        # Convert to grayscale image - "L" format means each pixel is represented by a single value from 0 to 255

        image_bw = im_pil.convert("L")
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        # Invert the image

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        # Converting to scalar quantity

        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

        # Using clip to limit the values between 0, 255

        image_bw_resized_inverted_scaled = np.flip(image_bw_resized_inverted-min_pixel,0, 255)
        max_pixel = np.max(image_bw_resized_inverted)

        # Converting to an array

        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        # Creating a test sample and making a prediction

        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is ", test_pred)

        # Display the resulting frame

        cv2.imshow("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        pass

# When everything is done, release the capture

cap.release()
cv2.destroyAllWindows()