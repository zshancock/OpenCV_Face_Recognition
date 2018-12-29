
## OpenCV_Face_Recognition (In progress)
#### Zac Hancock (zshancock@gmail.com)

### Introduction
The Haar cascades method of facial detection still works very well when compared to deep learning counterparts. OpenCV's library offers some good tools to get started detecting faces right away. However, I couldn't find a ton of resources on how to train a model to detect faces *I wanted it to detect* - that is actually recognize whose face it was. I compiled a mini-data set of images of Selena Gomez, Taylor Swift and Drake and set out to develop a recognizer that could not only identify a face, but label it as well. 


### Prerequisites

I ran this with Python 3.5 in a Juptyer notebook with OpenCV 3.3. Consult OpenCV documentation for installation instructions (https://opencv.org/). Other basic dependencies like numpy, matplotlib, etc were used. 

Basically, the following needs to run with no issues and you are all set. 
```
import cv2
```

### Running the Project

First you will need to run all the code in *setting_up_recognizer.py* - as it contains not only the necessary custom function **multiple_preds** but also contains the procedure to create your **trained_recognizer.yml**. This file can get very large, and therefore is not included in the project, but with your images, will apear in the project space in the **data** folder (where the haarcascades.xml file is). 

Once the *setting_up_recognizer.py* has been successful, the *running_recognizer.py* can be ran. Currently it is coded to call a single image from the **validation** folder. The validation folder has one sample image in it currently of Taylor Swift - called test1.jpg and test1_pred.jpg. This is an example of whats possible with the recognizer - a labeled face, not just a bounded face. 


### Discussion

...

