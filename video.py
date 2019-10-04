"""
Face identifier. To save a face, press S. 
To train the model so it can identify and show the faces saved, press T.
To quit program, press Q.
"""

# Imports
import numpy as np
import cv2
import os

# Global variables
LAST_NAME = ""
IS_SAVING_FACE = False
TRAINED = False
FRAMES_SAVED = 0
PERSONS = []
RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

# Create save function
def saveFace():
    global LAST_NAME, IS_SAVING_FACE
    LAST_NAME = input("Face owner name: ")
    print(LAST_NAME + "'s face is being saved.")
    IS_SAVING_FACE = True

def saveImg(img):
    if not os.path.exists("train"):
        os.makedirs("train")
    if not os.path.exists(f"train/{LAST_NAME}"):
        os.makedirs(f"train/{LAST_NAME}")
    files = os.listdir(f"train/{LAST_NAME}")
    cv2.imwrite(f"train/{LAST_NAME}/{str(len(files))}.jpg", img)

def trainData():
    global PERSONS, RECOGNIZER, TRAINED
    PERSONS = os.listdir("train")
    ids = []
    faces = []
    for i, person in enumerate(PERSONS):
        for file in os.listdir(f"train/{person}"):
            img = cv2.imread(f"train/{person}/{file}",0)
            faces.append(img)
            ids.append(i)
    RECOGNIZER.train(faces, np.array(ids))
    TRAINED = True

# Capture video file
# For webcam capture, enter: "0" or "1"
# For droidcam capture, enter something like http://localhost:8080/video
video_file = "../resources/joey_food.mp4"
video = cv2.VideoCapture(video_file)

# Read cascade classifier, that is going to classify something as a face
cascade_file = "../resources/haarcascade-frontalface-default.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)

# Create infinite loop
while(True):

    # Get frame
    ret, frame = video.read()

    if ret:
        
        # Create gray image/frame, we need it to work on recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Here we collect all the faces found on the image/frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Now we loop through it getting X,Y, width and height of our face "boxes"
        for (x,y,w,h) in faces:
            # Cut the faces
            roi = gray[y:y+h, x:x+w]

            # All faces must be 50x50
            roi = cv2.resize(roi, (50,50))

            #Then we draw the boxes around the faces
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)

            # Check if it's already trained, if so, show face names
            if TRAINED:
                face_id, conf = RECOGNIZER.predict(roi)
                person_name = PERSONS[face_id]
                cv2.putText(frame, person_name, (x+5,y-5), 3, 2, (255,0,255), 2, cv2.LINE_AA)

            # Check if it's saving
            if IS_SAVING_FACE:
                saveImg(roi)
                FRAMES_SAVED += 1

            # Limit the saved frames
            if FRAMES_SAVED > 50:
                IS_SAVING_FACE = False
                FRAMES_SAVED = 0

        # Here we show the image/frame on the "Video" window
        cv2.imshow("Video", frame)

    # Check if stop key was pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord("s"):
        saveFace()
    elif key == ord("t"):
        trainData()

# Free video cache
video.release()

# Destroy all open windows
cv2.destroyAllWindows()



