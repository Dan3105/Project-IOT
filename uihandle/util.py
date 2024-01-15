import os
import pickle
import io
import cv2

import tkinter as tk
from tkinter import messagebox

import firebase_admin
from firebase_admin import credentials, db
import requests

DOOR_CLOSED, DOOR_OPENED = 0, 1
BELL_OFF, BELL_ON = 0, 1
LED_OFF, LED_ON = 0, 1

# Set your bot token and chat ID
bot_token = "Your token ID"
chat_id = "Your chat ID"

url = 'http://192.168.0.114/cam-lo.jpg'

cred = credentials.Certificate("firebaseKey/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://thuchanh2-28511-default-rtdb.firebaseio.com/'
})
ref_door_mode = db.reference("door_mode")
ref_bell_mode = db.reference("bell_mode")
ref_led_mode = db.reference("led_mode")


def get_door_state():
    return ref_door_mode.get()


def set_door_state(mode=DOOR_CLOSED):
    ref_door_mode.set(mode)
    return


def get_bell_state():
    return ref_bell_mode.get()


def set_bell_state(mode=BELL_OFF):
    ref_bell_mode.set(mode)
    return


def get_led_state():
    return ref_led_mode.get()


def set_led_state(mode=LED_OFF):
    ref_led_mode.set(mode)
    return


def send_notification(_frame):
    _, img_encoded = cv2.imencode('.jpg', _frame)
    image_bytes = img_encoded.tobytes()

    # Create a file-like object
    image_file = io.BytesIO(image_bytes)

    files = {'photo': ('image.jpg', image_file)}  # Prepare the files parameter
    image_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto?chat_id={chat_id}"
    requests.post(image_url, files=files)

    # Send a text message
    text_message = "Strange human!"
    text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text_message}"
    requests.get(text_url)


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)
    