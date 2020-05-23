import cv2
import numpy as np
import dlib
from math import hypot

class VideoProcess:
    def __init__(self, casclf_path='./model/haarcascade_frontalface_default.xml'):
        self.faceCascade = cv2.CascadeClassifier(casclf_path)
        pass

    def color2gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        font = cv2.FONT_HERSHEY_COMPLEX
        faces = self.faceCascade(gray)
        return faces

