import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import cv2

def load_emotion_model(model_path):
  return torch.load(model_path)

def load_eye_model(model_path, some_params):
  eye_model = EyeModel(model_path)
  return eye_model

def save_model(model):
  pass

