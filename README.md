
## OpenCV_Face_Recognition (In progress)
#### Zac Hancock (zshancock@gmail.com)

### Introduction
The Haar cascades method of facial detection still works very well when compared to deep learning counterparts. OpenCV's library offers some good tools to get started detecting faces right away. However, I couldn't find a ton of resources on how to train a model to detect faces *I wanted it to detect* - that is actually recognize whose face it was. I compiled a mini-data set of images of Selena Gomez, Taylor Swift and Drake and set out to develop a recognizer that could not only identify a face, but label it as well. 

![alt text](https://github.com/zshancock/OpenCV_Face_Recognition/blob/master/graphics/validation_example.png)
**Raw unlabeled image -> Labeled image with indentified faces bound.**

If you have any large amount of images (with faces) of any individual, you can slightly alter this project to train a recognizer to identify any person(s). Currently, the **training_images** folder and the **extracted_faces** folder have example images inside them, but these alone would not be sufficient for training. I would recommend around 100+ images of a face at slightly different angles and lighting to build an appropriate model (that's is roughly what I had to train with Selena, Taylor and Drake). 

### Prerequisites

I ran this with Python 3.5 in a Juptyer notebook with OpenCV 3.3. Consult OpenCV documentation for installation instructions (https://opencv.org/). Other basic dependencies like numpy, matplotlib, etc were used. 

Basically, the following needs to run with no issues and you are all set. 
```
import cv2
```

### Running the Project

Getting acclimated with how images are expected in this program first means getting the naming convention. All images are placed in the training_image folder with this convention:
**Name.class.uniqueID.jpg** example: **selena.1.14.a.jpg**
*You can use any 'UniqueID' you want, but the 'class' must match what you assign in the 'people' list below.*

```
# Create the list for the model (must match the intended training model - i.e. if you have different individuals, the list needs to match)
people = ['Unknown', 'Selena Gomez', 'Taylor Swift', 'Drake']
colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)] # red, green, blue, yellow
```

First you will need to run all the code in *setting_up_recognizer.py* - as it contains not only the necessary custom function **multiple_preds** but also contains the procedure to create your **trained_recognizer.yml**. This file can get very large, and therefore is not included in the project, but with your images, will apear in the project space in the **data** folder (where the haarcascades.xml file is). 

Also inside the *setting_up_recognizer.py* file is the **extract_face** function (below). This function is used to extract the face from all the training images in the training_images folder. Use this in combination with the loop later in the program and it will process all the images in the training_images folder for your recognizer. The loop wherein I implement the extract_face function will automatically maintain the naming convention above but you may need to tweak it slightly to match the unique people your model is training for. (**Name.class.uniqueID.jpg**) 

```
def extract_face(image_path):
  ...
```

Multiple_preds has a few details that should be explained, esspecially if you are planning to deploy this on your own images. First, there is a 'confidence' metric. This is slightly counter-intuitive, as larger confidence means worse results. *Think of confidence in as distance **away** from the actual...larger numbers mean farther away from a perfect prediction.* I currently have the confidence threshold at 60 ~ which means, if the confidence is worse than 60, the recognizer will label the face 'Unknown'. As mentioned before, if you have a different list of individuals you are identifying (i.e. Instead of unknown, selena, taylor, drake you have unkown, Alex, Aaron, Adam), you need to alter the 'people' list.  

```
def multiple_preds(img):
  ...
  # Confidence in this case is 0-200 and is "distance" from the correct prediction. For example
        # confidence of 200 is awful/random/unknown while a prediction of 10 is virtually maximum confidence.
        # Research suggested setting this to 60 and proved to be effective - anything over 60 will be identified 
        # as unknown. 
        
        # Set the name according to id
        if person == 1 and conf < 60:
            color = colors[1]
            person = people[1]
            # Put text describe who is in the picture
        elif person == 2 and conf < 60:
            color = colors[2]
            person = people[2]
            # Put text describe who is in the picture
        elif person == 3 and conf < 60:
            color = colors[3]
            person = people[3]
        else:
            color = colors[0]
            person = people[0]
            
        draw_rectangle(img, (x,y,w,h), color)
        draw_text(img, person, x, y, color)
```


Once the *setting_up_recognizer.py* has been successful, the *running_recognizer.py* can be ran. Currently it is coded to call a single image from the **validation** folder. The validation folder has one sample image in it currently of Taylor Swift - called test1.jpg and test1_pred.jpg. This is an example of whats possible with the recognizer - a labeled face, not just a bounded face. 


### Discussion

In the end, my recognizer, though yours may vary depending on your image data set, was *average* at recognizing specific faces. Like below, it performed well in some cases (*as mentioned before, I had around 100 images of each Selena Gomez, Taylor Swift and Drake from Google Images*). However, the Haar cascades have difficutly if the face is not fully visible, at a weird angle, etc. A **deep learning** approach may be more suitable, however I will admit this method worked quickly and was fairly straight forward with OpenCV documentation. Training the recognizer required a couple custom built functions, but the recognizer itself and the face detection (cascades) was only slightly modified from the defaults in the OpenCV library. 


![alt text](https://github.com/zshancock/OpenCV_Face_Recognition/blob/master/graphics/validation_example2.png)
**Correctly identified Taylor Swift (left) and Selena Gomez (right) and 2 'unknowns' were detected in audience.**


Next steps would be perhaps incorporating a deep learning procedure for identifying the face, after Haar cascades method has found a face ~ or going to a different detection method to account for faces not head on. OpenCV has a lot to offer, and I will continue to explore it as I build my image proccessing, object detection, image classification, etc. skillset. 
