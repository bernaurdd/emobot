import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2

# --- functions ---
from src.adviser import Adviser
from src.detection.detection import EmotionDetection

detection = EmotionDetection('detection')
emotion = None
adviser = Adviser()


def update_frame():
    ret, frame = cap.read()
    image = Image.fromarray(frame[:, :, ::-1])
    photo.paste(image)
    global emotion
    # description['text'] = 'new text'
    detection_response = detection.detect_emotion(image)
    if detection_response is not None:
        print(detection_response)
        emotion = detection_response
    root.after(20, update_frame)


def send():
    user_input = user.get()
    topic = adviser.predictTopic(user_input)
    bot.set(adviser.getAdvice(emotion, topic))
    user_entry.delete(0, 'end')


def detect():
    prompt = adviser.getPrompt(emotion)
    bot.set(prompt)


cap = cv2.VideoCapture(0)

ret, frame = cap.read()

root = tk.Tk()

image = Image.fromarray(frame)
photo = ImageTk.PhotoImage(image)

user = StringVar()
bot = StringVar()
user_entry = tk.Entry(root, textvariable=user, width=100)
user_entry.pack(side='bottom')

bot_entry = tk.Entry(root, textvariable=bot, width=100)
bot_entry.pack(side='bottom')
bot.set('I am detecting your mood wait 5 seconds and push the button')

speak_button = tk.Button(root, text="speak", command=send)
speak_button.pack(side='bottom')

detect_button = tk.Button(root, text="detect mood", command=detect)
detect_button.pack(side='bottom')

ask = ["hi", "hello"]
hi = ["hi", "hello", "Hello too"]
error = ["sorry, i don't know", "what u said?"]

canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
canvas.pack(side='bottom', fill='both', expand=True)

canvas.create_image((0, 0), image=photo, anchor='nw')

update_frame()

root.mainloop()

cap.release()
