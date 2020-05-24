import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
from model import EyeModel
from model import EmotionModel

class Model:
  def __init__(self, eye_model_path, emotion_model_path):
    self.eye_model = None
    self.emotion_model = None
    if eye_model_path is not None:
      self.load_eye_model(eye_model_path)
    if emotion_model_path is not None:
      self.load_emotion_model(model_path=emotion_model_path)

  def load_emotion_model(self, model_path):
    self.emotion_model = EmotionModel.EmotionModel(model_path=model_path)

  def load_eye_model(self, model_path):
    self.eye_model = EyeModel.EyeModel(model_path)
