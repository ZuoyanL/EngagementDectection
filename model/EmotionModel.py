import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import numpy as np
import cv2

class EmotionModel:
  def __init__(self, in_channel=1, in_width=64, in_weight=64, model_path=""):
    self.model = torch.load(model_path)
    self.frame_count = 0
    self.emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
    self.in_channel = in_channel
    self.in_width = in_width
    self.in_weight = in_weight
  
  def detect_emotion(self, faces, frame, in_channel=1):
    if len(faces) > 0:
      for x, y, width, height in faces:
        if in_channel == 1:
          # 如果模型的输入是单通道的
          cropped_face = frame[y:y+height, x:x+width] # 把脸给动态滴截取下来
        if in_channel == 3:
          # 如果模型的输入是3通道的
          cropped_face = frame[:, y:y+height, x:x+width]
        test_image = cv2.resize(cropped_face, (self.in_width, self.in_weight))
        test_image = test_image.reshape([-1, self.in_width, self.in_weight, in_channel])

        test_image = np.multiply(test_image, 1.0 / 255.0)
        
        probab = self.model(test_image)[0] * 100
        label = np.argmax(probab)
        probab_predicted = int(probab[label])
        predicted_emotion = self.emotions[label]
        self.frame_count = 0
        self.emotion = label
    self.frame_count += 1
    return self.emotion
        


    
