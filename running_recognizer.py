'''
Run the setting_up_recognizer.py file first - the training protocol (.yml file) should be in the directory in the 'data' folder. The .yml
for the training procedure is quite large (~50MB) and therefore cannot be added to my repository. Running on your local machine using the
procedure in setting_up_recognizer.py will generate one on your local machine (data/train_recongizer.yml). 

This step will require pathing to a validation image, if the entire project was 
cloned/copied, then there is one image of Taylor Swift called 'test1' in the validation folder. (as coded below). 
Additionally, multiple_preds is called from the setting_up_recognizer.py file, so it must be in the directory.

''' 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline # if preference is displaying images in python console, rather than with cv2 package.

from PIL import Image
from setting_up_recognizer import multiple_preds


face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("data/train_recognizer.yml")


print("Predicting images...")

#load test images
test_img1 = cv2.imread("validation/test1.jpg")


#perform a prediction
predicted_img1 = multiple_preds(test_img1)
print("Prediction complete")

#save the predicted output.
cv2.imwrite('validation/test1_pred.jpeg', predicted_img1)
