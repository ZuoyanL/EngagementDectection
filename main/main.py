import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model
from Eyes import get_blinking_ratio, get_gaze_ratio
from Emotion import detect_emotion
from google.colab.patches import cv2_imshow


# Function for finding midpoint of 2 points
def midpoint(p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

    

def gen_concentration_index(emotion,size,x):
        weight = 0
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6, 3: 0.3, 4: 0.6, 5: 0.9}
        gaze_weights = 0
        if size < 0.2:
            gaze_weights = 0
        elif size > 0.2 and size < 0.3:
            gaze_weights = 1.5
        else:
            if x < 2 and x > 1:
                gaze_weights = 5
            else:
                gaze_weights = 2

        concentration_index = (emotionweights[emotion] * gaze_weights) / 4.5
        if concentration_index > 0.65:
            return "You are highly engaged!"
        elif concentration_index > 0.25 and concentration_index <= 0.65:
            return "You are engaged."
        else:
            return "Pay attention!"


    # Main function for analysis


def detect( frame,predictor,detector,faceCascade,emotion_model):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = detector(gray)

        for face in faces:    #对我们来说就是单脸
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            f = gray[x:x1, y:y1]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = predictor(gray, face)
            left_point = (landmarks.part(36).x, landmarks.part(36).y)
            right_point = (landmarks.part(39).x, landmarks.part(39).y)
            center_top = midpoint(landmarks.part(37), landmarks.part(38))
            center_bottom = midpoint(landmarks.part(41), landmarks.part(40))
            hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
            ver_line = cv2.line(frame, center_top,center_bottom, (0, 255, 0), 2)
            
            # Eyedetecion
            left_eye_ratio = get_blinking_ratio(frame,[36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_lr, gaze_ratio_ud = get_gaze_ratio(frame,[36, 37, 38, 39, 40, 41], landmarks, gray)

            # Emotion detection
            emotion = detect_emotion(gray,emotion_model,faceCascade)
            
            # add text
            ci = gen_concentration_index(emotion,gaze_ratio_lr,left_eye_ratio)
            emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
            cv2.putText(frame, emotions[emotion],(50, 150), font, 2, (0, 0, 255), 3)
            cv2.putText(frame, ci,(50, 250), font, 2, (0, 0, 255), 3)
        return frame

# load 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../util/model/shape_predictor_68_face_landmarks.dat")
emotion_model = load_model('../util/model/emotion_recognition.h5')  #from zuoyan
faceCascade = cv2.CascadeClassifier('../util/model/haarcascade_frontalface_default.xml')

# load data
frame = cv2.imread('/content/distractionModel/Data/BEN.jpg')  # from xiaotong

#detect
reframe = detect(frame,predictor,detector,faceCascade,emotion_model)

cv2_imshow(reframe)