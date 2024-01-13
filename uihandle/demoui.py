import os.path
import datetime
import pickle
import time
import tkinter as tk
import urllib.request
from io import BytesIO
import requests

import numpy as np

import cv2
from PIL import Image, ImageTk
import util
import torch
from model_handler import ModelAntiSpoffing, ModelRecognition
import os
from tkinter import messagebox

CRR_PATH = os.curdir

MODEL_RECOG_PATH = 'model-2/face_recognizer_fast.onnx'
MODEL_DETECT_PATH = 'model-2/yunet_s_640_640.onnx'
MODEL_ANTI_PATH = 'antispoof.pt'

DB_IMAGE_PATH = 'db/image-data'
DB_CSV_PATH = 'db/db.csv'

url = 'http://192.168.0.114/cam-lo.jpg'
IS_REGISTER=False
class App:
    def __init__(self):
        self.allow_detect = True

        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520")
        # self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        # self.login_button_main_window.place(x=750, y=200)

        # self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        # self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.anti_spoof_model = ModelAntiSpoffing(MODEL_ANTI_PATH)
        self.model_recog = ModelRecognition(MODEL_DETECT_PATH, MODEL_RECOG_PATH, DB_CSV_PATH, DB_IMAGE_PATH)

        self.add_webcam(self.webcam_label)


        # self.db_dir = './db'
        # if not os.path.exists(self.db_dir):
        #     os.mkdir(self.db_dir)

        # self.log_path = './log.txt'

    def add_webcam(self, label):
        # phan nay nhan cam
        self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        #ret, frame = self.cap.read()
        #if frame is not None:
        #ret, frame = self.cap.read()
        #print(frame.shape)

        response = requests.get(url)
        frame = np.array(Image.open(BytesIO(response.content)))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(frame.shape)

        if IS_REGISTER:
            return
        if frame is not None:

            #Handle logic cua mo hay dong

            #Mat real, Mat fake
            image_moded, face = self.anti_spoof_model.detect(frame)
            if face is not None:
                #neu mat real
                self.most_recent_capture_arr = face
                img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
                self.most_recent_capture_pil = Image.fromarray(img_)

                imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
                self._label.imgtk = imgtk
                self._label.configure(image=imgtk)

                #ten nguoi dung
                result_name = self.model_recog.predict(face)
                if result_name is not None:
                    # lam gi thi lam
                    print(result_name)
    
        self._label.after(20, self.process_webcam)

    def login(self):
        pass

    def logout(self):
        pass

    def validate_text(self, event):
        start = "1.0"
        end = "end-1c"
        text_content = self.entry_text_register_new_user.get(start, end)
        print(text_content)
        if text_content and not text_content[-1].isalnum():
            messagebox.showwarning("Invalid Entry", "Only allow letters and numbers")

            # Remove the last entered character (punctuation)
            self.entry_text_register_new_user.delete("end-2c")

    def on_closing(self, event):
        self.main_window.deiconify()
        self.allow_detect = True

    def on_openning(self, event):
        self.main_window.withdraw()
        self.allow_detect = False

    def register_new_user(self):
        IS_REGISTER = True
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.register_new_user_window.bind("<Visibility>", self.on_openning)
        self.register_new_user_window.bind("<Destroy>", self.on_closing)
        #self.register_new_user_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green',
                                                                      self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again',
                                                                         'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.bind('<KeyRelease>', self.validate_text)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window,
                                                                'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        try:
            _, face_human = self.anti_spoof_model.detect(self.most_recent_capture_pil)
            if face_human is not None:
                self.model_recog.save_data_user(face_human, name)
                util.msg_box('Success!', 'User was registered successfully !')
        except Exception as e:
            print(e)
        IS_REGISTER = False
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
