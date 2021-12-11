import tkinter as tk
from PIL import Image, ImageTk
import cv2

# --- functions ---

def update_frame():

    ret, frame = cap.read()

    image = Image.fromarray(frame)
    photo.paste(image)

    #description['text'] = 'new text'

    root.after(10, update_frame) # update it again after 10ms

# --- main ---

cap = cv2.VideoCapture(0)

# get first frame
ret, frame = cap.read()

# - GUI -

root = tk.Tk()

image = Image.fromarray(frame)
photo = ImageTk.PhotoImage(image)  # it has to be after `tk.Tk()`

canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
canvas.pack(side='left', fill='both', expand=True)

canvas.create_image((0,0), image=photo, anchor='nw')

description = tk.Label(root, text="Place for description")
description.pack(side='right')

# - start -

update_frame() # update it first time

root.mainloop() # start program - this loop runs all time

# - after close -

cap.release()