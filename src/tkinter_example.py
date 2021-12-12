import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from random import choice
import time
# --- functions ---
from src.detection.detection import EmotionDetection

detection = EmotionDetection('detection')

def update_frame():
    ret, frame = cap.read()
    image = Image.fromarray(frame[:, :, ::-1])
    photo.paste(image)
    global emotion
    #description['text'] = 'new text'
    emotion = detection.detect_emotion(image)
    root.after(20, update_frame)


def send():
    question = user.get()
    if question in ask:
        bot.set(choice(hi))
        user_entry.delete(0, 'end')
    else:
        bot.set(choice(error))
        user_entry.delete(0, 'end')

def detect():
    bot.set(emotion)



cap = cv2.VideoCapture(0)


ret, frame = cap.read()

root = tk.Tk()

image = Image.fromarray(frame)
photo = ImageTk.PhotoImage(image)

user = StringVar()
bot = StringVar()
user_entry = tk.Entry(root, textvariable=user,width=100)
user_entry.pack(side='bottom')

bot_entry = tk.Entry(root, textvariable=bot,width=100)
bot_entry.pack(side='bottom')

speak_button = tk.Button(root, text="speak", command=send)
speak_button.pack(side='bottom')

detect_button = tk.Button(root, text="detect mood", command=detect)
detect_button.pack(side='bottom')

ask   = ["hi", "hello"]
hi    = ["hi", "hello", "Hello too"]
error = ["sorry, i don't know", "what u said?" ]

canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
canvas.pack(side='bottom', fill='both', expand=True)

canvas.create_image((0,0), image=photo, anchor='nw')

update_frame()

root.mainloop()

cap.release()