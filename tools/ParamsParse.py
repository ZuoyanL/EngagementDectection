import argparse

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="progrom description")
        self.parser.add_argument('--video_path', type=str, help="video path", required=True)
        self.parser.add_argument('--cascade_model_path', type=str, default="./model/haarcascade_frontalface_default.xml")
        self.parser.add_argument('--emotion_model_path', type=str, default="./model/emotion_recognition.t7")
        self.parser.add_argument('--predictor_path', type=str, default="./model/shape_predictor_68_face_landmarks.dat")
        self.parser.add_argument('--result_path', type=str, default="../results/", required=True)
        self.args = self.parser.parse_args()
