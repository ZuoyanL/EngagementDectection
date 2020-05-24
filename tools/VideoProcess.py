import cv2
import numpy as np
import dlib
from math import hypot

class VideoProcess:
    def __init__(self, casclf_path='../model/haarcascade_frontalface_default.xml'):
        self.faceCascade = cv2.CascadeClassifier(casclf_path)

    # 彩色转灰度
    def color2gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # 把一帧图片中的所有人脸都从检测出来，并返回人脸坐标的集合，不能打乱顺序
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        faces = self.faceCascade.detectMultiScale(gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        return faces

    def display_results(self, frame, faces, CIs, emotions, rules={'good':0.65,
                                                        'normal':0.25}):
        """
        :param frame: 每一帧
        :param faces: 每一帧里面出现的人脸
        :param CI: 人脸对应的CI分数
        :param rules: 几个阈值，默认为 (0, 0.25] (0.25, 0.65] (0.65, +)
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        i = 0
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            i += 1
            if CIs[i] >= rules['good']:
                color = (0, 255, 0)
            elif CIs[i] >= rules['normal']:
                color = (255, 255, 255)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
            cv2.putText(frame, CIs[i], (50, 250), font, 2, color, 3)
        return frame
