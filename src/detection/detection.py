import time

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import os

from src.adviser import Adviser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl

mpl.use('TkAgg')


class EmotionDetection:
    def __init__(self, folder_path):
        # Create the model
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        self.model = model
        self.model.load_weights(f'{folder_path}/model.h5')

        self.emobot = Adviser()
        self.folder_path = folder_path
        self.start_time = time.perf_counter()
        self.emotion_count = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "neutral": 0, "sad": 0, "surprise": 0}

    def start_detection(self):
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        quote = None
        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
        emotion_count = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "neutral": 0, "sad": 0, "surprise": 0}

        # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            width = int(cap.get(3))
            height = int(cap.get(4))
            font = cv2.FONT_HERSHEY_SIMPLEX

            facecasc = cv2.CascadeClassifier(f'{self.folder_path}/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = self.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                # cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if emotion_dict[maxindex] in emotion_count:
                    emotion_count[emotion_dict[maxindex]] += 1

            if quote is not None:
                cv2.putText(frame, f'{quote}', (width // 2, height // 2), font, 1, (0, 0, 0), 5, cv2.LINE_AA)

            if time.perf_counter() - self.start_time > 10:
                self.start_time = time.perf_counter()
                emotion = max(emotion_count, key=emotion_count.get)
                quote = self.emobot.getAdvice(emotion)

                print(self.emobot.getAdvice(emotion))

                emotion_count = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "neutral": 0, "sad": 0, "surprise": 0}

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _preprocess_image(self, image):
        return np.array(image)

    def detect_emotion(self, image):
        image = self._preprocess_image(image)
        emotion_dict = {0: "anger", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

        facecasc = cv2.CascadeClassifier(f'{self.folder_path}/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            # cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if emotion_dict[maxindex] in self.emotion_count:
                self.emotion_count[emotion_dict[maxindex]] += 1

        if time.perf_counter() - self.start_time > 10:
            self.start_time = time.perf_counter()
            emotion = max(self.emotion_count, key=self.emotion_count.get)
            self.emotion_count = {"anger": 0, "disgust": 0, "fear": 0, "happy": 0, "neutral": 0, "sad": 0, "surprise": 0}
            return emotion
            # print(self.emobot.getAdvice(emotion))
        return None
