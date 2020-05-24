import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
import cv2
import model.VGG as VGG
import torchvision.transforms as transforms
from PIL import Image

class EmotionModel:
  def __init__(self, in_channel=1, in_width=48, in_weight=48, model_path="abc"):
    self.model = VGG.VGG('VGG19')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    self.model.load_state_dict(checkpoint['net'])
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    #self.model.to(device)
    self.frame_count = 0
    self.emotions = {0: 'Angry', 1:'Disgust', 2: 'Fear', 3: 'Happy',
                    4: 'Sad', 5: 'Surprised', 6: 'Neutral'}
    self.in_channel = in_channel
    self.in_width = in_width
    self.in_weight = in_weight
  
  def detect_emotion(self, face, frame, in_channel=1):
    x, y = face.left(), face.top()
    x1, y1 = face.right(), face.bottom()
    if in_channel == 1:
      # 如果模型的输入是单通道的
      cropped_face = frame[y:y1, x:x1] # 把脸给动态滴截取下来
    if in_channel == 3:
      # 如果模型的输入是3通道的
      cropped_face = frame[:, y:y1, x:x1]

    test_image = cv2.resize(cropped_face, (self.in_width, self.in_weight))
    cut_size = 44
    test_image = test_image[:, :, np.newaxis]
    test_image = np.concatenate((test_image, test_image, test_image), axis=2)
    test_image = Image.fromarray(test_image)
    transform_test = transforms.Compose([
      transforms.TenCrop(cut_size),
      transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])
    #test_image = test_image.reshape([-1, in_channel, self.in_weight, self.in_width])

    inputs = transform_test(test_image)
    inputs = Variable(inputs, volatile=True)
    self.model.eval()
    outputs = self.model(inputs)[0] * 100
    outputs_avg = outputs.view(1, -1).mean(0)
    score = F.softmax(outputs_avg)
    _, predicted = torch.max(outputs_avg.data, 0)
    self.frame_count = 0
    self.emotion = int(predicted.numpy())
    print(self.emotions[self.emotion])
    return self.emotion
        


    
