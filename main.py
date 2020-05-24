"""
procedure of whole algorithm
"""
import cv2
import dlib
import numpy
import torch
from model import Model
from tools import VideoProcess
import tools

# main function.
def main(params):
    eye_model_path = params.predictor_path
    emotion_model_path = params.emotion_model_path
    print(emotion_model_path)
    cascade_model_path = params.cascade_model_path
    result_path = params.result_path
    video_path = params.video_path
    model = Model.Model(eye_model_path, emotion_model_path)
    video_process = VideoProcess.VideoProcess(casclf_path=cascade_model_path)

    cap = cv2.VideoCapture(video_path)

    while(1):
        ret, frame = cap.read()
        print(frame is None)
        cv2.imshow("original", frame)
        faces = video_process.detect_faces(frame)
        emotions, CIs = detect(frame, faces, model.eye_model,
                               model.emotion_model, video_process)
        for i in CIs:
            print(i)
        video_process.display_results(frame, faces, CIs, emotions)
        # if in notebook, using
        # cv2_imshow(reframe)
        cv2.imshow("video", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# calculate the CI.
def gen_concentration_index(emotion, size, x):
    weight = 0
    emotionweights = {0: 0.2, 1: 0.25, 2: 0.3, 3: 0.6, 4: 0.3, 5: 0.6, 6:0.9}
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
    return concentration_index

# Function for finding midpoint of 2 points
def midpoint(p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def detect(frame, faces, eye_model, emotion_model, video_process):
    gray = video_process.color2gray(frame)
    emotions = []
    CIs = []
    for face in faces:  # 对我们来说就是单脸
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        f = gray[x:x1, y:y1]
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = eye_model.predictor(gray, face)
        left_point = (landmarks.part(36).x, landmarks.part(36).y)
        right_point = (landmarks.part(39).x, landmarks.part(39).y)
        center_top = eye_model.midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom = eye_model.midpoint(landmarks.part(41), landmarks.part(40))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

        # Eyedetecion
        left_eye_ratio = eye_model.get_blinking_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_lr, gaze_ratio_ud = eye_model.get_gaze_ratio(frame, [36, 37, 38, 39, 40, 41], landmarks, gray)

        # Emotion detection
        emotion = emotion_model.detect_emotion(face, gray)
        emotions.append(emotion)
        # add text
        ci = gen_concentration_index(emotion, gaze_ratio_lr, left_eye_ratio)
        CIs.append(ci)
    return emotions, CIs

import sys
from tools import ParamsParse
if __name__ == '__main__':
    parser = ParamsParse.Parser()
    params = parser.args
    main(params)