''' 
Pipeline Step - Image Training  and Recognition

Extract faces from the 'training_images' folder, place into the 'extracted_faces' folder. Use 'extracted_faces' to 
train the recognizer. Deploy the Local Binary Patterns recognizer (which is considered faster, even if slightly 
less effective at locating faces). Save the trained model parameters in 'data/models'. Deploy on images in 
'05_validation' folder. 

Note: The extracted face function will generate a corrupted .jpg to the 'extracted_face' folder if no face was detected in the 
augmented image. These need to be manually accounted for, or additional loop needs added for 'No Faces Detected.'

'''


# Import dependencies

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline # if preference is displaying images in python console, rather than with cv2 package.

from PIL import Image

'''
Create Helpful Functions
Random tasks that opencv commands sometime require a lot of typing - 
simplify some of the steps with custom functions (like changing color to gray, etc.)
'''

def convertcolors(image): 
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert2gray(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def draw_rectangle(img, rect, a):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), a, 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y, a):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, a, 1)


'''
Create more complicated functions necessary for processing images
These are functions wherein the analyst had to make without direction ~ few tutorials exist 
beyond the OpenCV documentation. Essentially, the analyst needed a way to extract the face 
from augmented images for training & predict multiple faces on a single screen/image (and identify 
the detected face based on confidence).
'''

def extract_face(image_path):
    haar = "data/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(haar)

    img = cv2.imread(image_path)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = face_cascade.detectMultiScale(miniframe, 1.4, 5)
    
    if len(faces) == 0:
        return None
    
    else:
        for (x,y,w,h) in faces:
        
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))
            sub_face = img[y:y + h, x:x + w]
              
            return(sub_face)


def multiple_preds(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    # For each face in faces, we will start predicting using pre trained model
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(gray, (x, y), (x+w, y+h), (0,255,0), 4)

        # Recognize the face belongs to which ID
        person, conf = face_recognizer.predict(gray[y:y+h,x:x+w]) 

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
        
    return img


# Create the list for the model (must match the intended training model - i.e. if you have different individuals, the list needs to match)
people = ['Unknown', 'Selena Gomez', 'Taylor Swift', 'Drake']
colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)] # red, green, blue, yellow


'''
Extract faces for training from the training images using above function.
'''

trainingimages = os.listdir('training_images')

# these are counters for to ensure unique file names for each person
selena = 1
taylor = 1
drake = 1
unknown = 1

for image_path in trainingimages:

# determine the image counter
    if 'selena' in image_path:
        counter = selena
        path = 'extracted_faces/selena.1.'
    elif 'taylor' in image_path:
        counter = taylor
        path = 'extracted_faces/taylor.2.'
    elif 'drake' in image_path:
        counter = drake
        path = 'extracted_faces/drake.3.'
    else:
        counter = unknown
        path = 'extracted_faces/unkown.4.'
    
    savefile = path + str(counter) + '.jpg'
    imgpath = 'training_images/' + image_path
    cv2.imwrite(savefile, extract_face(imgpath))

    
    # add one to the counter, depending on who was the name.
    if 'selena' in image_path:
        selena +=1
    elif 'taylor' in image_path:
        taylor +=1
    elif 'drake' in image_path:
        drake +=1
    else:
        unknown +=1


'''
Set up a training procedure for the recognizer to known which 'label' each face corresponds to.
'''
def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs,faces


# Train the recognzier. Save a yml with training protocol to retrieve during implementation. 

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='04_extracted_faces'

Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('data/train_recognizer.yml')
cv2.destroyAllWindows()




