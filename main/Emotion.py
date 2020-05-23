from keras.models import load_model
import cv2
import numpy as np

def detect_emotion(gray,emotion_model,faceCascade):
        # Dictionary for emotion recognition model output and emotions
        frame_count = 0
        emotion = 5
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=7,minSize=(100, 100))

        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height, x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1, 48, 48, 1])
                test_image = np.multiply(test_image, 1.0 / 255.0)
                
                if frame_count % 5 == 0:
                    probab = emotion_model.predict(test_image)[0] * 100
                    label = np.argmax(probab)
                    probab_predicted = int(probab[label])
                    predicted_emotion = emotions[label]
                    frame_count = 0
                    emotion = label

        frame_count += 1
        return emotion