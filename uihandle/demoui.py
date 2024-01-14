import os.path
import datetime
import pickle
import time
import tkinter
import tkinter as tk
import urllib.request
from io import BytesIO
import requests
from datetime import datetime, timedelta
from tkinter import *
from tkinter import ttk
import numpy as np
import pandas as pd
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
IS_REGISTER = False

is_ringing = False
is_opening = False


class App:
    def __init__(self):
        self.allow_detect = True
        self.first_time_discovering_a_person_without_permission = -1
        self.last_time_discovering_a_person_without_permission = -1
        self.first_time_discovering_a_person_with_permission = -1
        self.last_time_discovering_a_person_with_permission = -1
        self.last_time_allow_open_door = datetime(1970, 1, 1)
        self.last_time_allow_alarm = datetime(1970, 1, 1)
        self.warning_wait_time = 5
        self.clear_wait_time = 7
        self.open_door_wait_time = 3
        self.alarm_time = 5
        self.open_door_time = 7
        self.photos = list()
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520")
        self.df = pd.read_csv(DB_CSV_PATH)

        # self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        # self.login_button_main_window.place(x=750, y=200)

        # self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        # self.logout_button_main_window.place(x=750, y=300)

        self.manage_permission_user_button_main_window = util.get_button(self.main_window, 'Manage permission list',
                                                                         'gray',
                                                                         self.manage_permission_list, fg='black')
        self.manage_permission_user_button_main_window.place(x=750, y=200)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.anti_spoof_model = ModelAntiSpoffing(MODEL_ANTI_PATH)
        self.model_recog = ModelRecognition(MODEL_DETECT_PATH, MODEL_RECOG_PATH, DB_CSV_PATH, DB_IMAGE_PATH)

        self.add_webcam(self.webcam_label)
        self.last_capture_face = None

        # self.db_dir = './db'
        # if not os.path.exists(self.db_dir):
        #     os.mkdir(self.db_dir)

        # self.log_path = './log.txt'

    def add_webcam(self, label):
        # phan nay nhan cam
        self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def get_diff_time(self, time_a, time_b):
        if type(time_a) == int or type(time_b) == int:
            return -1
        delta = time_a - time_b
        # print(delta.seconds)
        return delta.seconds

    def process_webcam(self):
        global is_ringing, is_opening
        ret, frame = self.cap.read()
        # if frame is not None:
        # ret, frame = self.cap.read()
        # print(frame.shape)

        # response = requests.get(url)
        # frame = np.array(Image.open(BytesIO(response.content)))
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(frame.shape)
        if is_ringing:
            print("bell is ringing")
        else:
            print("bell is not ringing")
        if is_opening:
            print("door is opening")
        else:
            print("door is closing")
        if IS_REGISTER:
            return
        if frame is not None:

            # Handle logic cua mo hay dong

            # Mat real, Mat fake
            image_moded, face = self.anti_spoof_model.detect(frame)
            # print(self.first_time_discovering_a_person_without_permission, self.last_time_discovering_a_person_without_permission)
            if self.get_diff_time(datetime.now(),
                                  self.last_time_discovering_a_person_without_permission) > self.clear_wait_time:
                self.first_time_discovering_a_person_without_permission = self.last_time_discovering_a_person_without_permission = -1

            if self.get_diff_time(datetime.now(),
                                  self.last_time_discovering_a_person_with_permission) > self.clear_wait_time:
                self.first_time_discovering_a_person_with_permission = self.last_time_discovering_a_person_with_permission = -1

            if is_opening == False and is_ringing == False and self.get_diff_time(datetime.now(),
                                                                                  self.last_time_allow_alarm) <= self.alarm_time:
                is_ringing = True
            if is_ringing == True and self.get_diff_time(datetime.now(),
                                                         self.last_time_allow_alarm) > self.alarm_time:
                is_ringing = False

            if is_ringing == False and is_opening == False and self.get_diff_time(datetime.now(),
                                                                                  self.last_time_allow_open_door) <= self.open_door_time:
                is_opening = True
            if is_opening == True and self.get_diff_time(datetime.now(),
                                                         self.last_time_allow_open_door) > self.open_door_time:
                is_opening = False
            if face is not None:
                # neu mat real
                # self.register_new_user_button_main_window.config(state=tk.NORMAL)
                self.register_new_user_button_main_window['state'] = tk.NORMAL
                self.last_capture_face = face
                # ten nguoi dung
                result_name = self.model_recog.predict(face)
                if result_name is not None:
                    print("ok")
                    if self.first_time_discovering_a_person_with_permission == -1:
                        self.first_time_discovering_a_person_with_permission = datetime.now()
                    else:
                        self.last_time_discovering_a_person_with_permission = datetime.now()
                        if self.get_diff_time(self.last_time_discovering_a_person_with_permission,
                                              self.first_time_discovering_a_person_with_permission) > self.open_door_wait_time:
                            self.last_time_allow_open_door = datetime.now()
                else:
                    print("not ok")
                    if self.first_time_discovering_a_person_without_permission == -1:
                        self.first_time_discovering_a_person_without_permission = datetime.now()
                    else:
                        self.last_time_discovering_a_person_without_permission = datetime.now()
                        if self.get_diff_time(self.last_time_discovering_a_person_without_permission,
                                              self.first_time_discovering_a_person_without_permission) > self.warning_wait_time:
                            self.last_time_allow_alarm = datetime.now()
            else:
                # self.register_new_user_button_main_window.config(state=tk.DISABLED)
                self.register_new_user_button_main_window['state'] = tk.DISABLED
                self.last_capture_face = None

            self.most_recent_capture_arr = image_moded
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)

            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        pass

    def logout(self):
        pass

    def validate_text(self, event):
        start = "1.0"
        end = "end-1c"
        text_content = self.entry_text_register_new_user.get(start, end)
        # print(text_content)
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
        if self.last_capture_face is None:
            return
        IS_REGISTER = True

        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.register_new_user_window.bind("<Visibility>", self.on_openning)
        self.register_new_user_window.bind("<Destroy>", self.on_closing)
        # self.register_new_user_window.protocol("WM_DELETE_WINDOW", self.on_closing)
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



    def manage_permission_list(self):
        self.manage_permission_list_window = tk.Toplevel(self.main_window)
        self.manage_permission_list_window.geometry("1200x520+370+120")
        self.manage_permission_list_window.title("Manage Permission List")

        self.manage_permission_list_main_frame = Frame(self.manage_permission_list_window)
        self.manage_permission_list_main_frame.pack(fill=BOTH, expand=1)

        self.manage_permission_list_canvas = Canvas(self.manage_permission_list_main_frame)
        self.manage_permission_list_canvas.pack(side=LEFT, fill=BOTH, expand=1)

        self.manage_permission_list_scrollbar = ttk.Scrollbar(self.manage_permission_list_main_frame, orient=VERTICAL,
                                                              command=self.manage_permission_list_canvas.yview)
        self.manage_permission_list_scrollbar.pack(side=RIGHT, fill=Y)

        self.manage_permission_list_canvas.configure(yscrollcommand=self.manage_permission_list_scrollbar.set)
        self.manage_permission_list_canvas.bind('<Configure>', lambda e: self.manage_permission_list_canvas.configure(
            scrollregion=self.manage_permission_list_canvas.bbox("all")))

        self.manage_permission_list_show_frame = Frame(self.manage_permission_list_canvas)

        self.manage_permission_list_canvas.create_window((0, 0), window=self.manage_permission_list_show_frame, anchor="nw")
        self.label_list = list()
        self.manage_permission_list_reload_window()

    def get_btn(self, index):
        self.df.iloc[index, 2] = 1 - self.df.iloc[index, 2]
        self.df.to_csv(DB_CSV_PATH, index=False)

    def delete_db(self, index):
        image_url = self.df.iloc[index, 1]
        os.remove(DB_IMAGE_PATH + '/' + image_url)
        self.df = self.df.drop(self.df.index[index])
        self.df.to_csv(DB_CSV_PATH, index=False)
        for label in self.label_list:
            label.destroy()
        self.label_list = list()
        self.manage_permission_list_reload_window()

    def manage_permission_list_reload_window(self):
        global photos
        self.df = pd.read_csv(DB_CSV_PATH)
        photos = [ImageTk.PhotoImage(Image.open(DB_IMAGE_PATH + '/' + img_url)) for img_url in
                  self.df[self.df.columns[1]].values.tolist()]
        print(photos)
        for row in range(len(photos)):
            name = self.df.iloc[row, 0]
            permission = self.df.iloc[row, 2]
            photo_label = Label(self.manage_permission_list_show_frame, image=photos[row])
            photo_label.grid(row=row, column=1)
            name_label = Label(self.manage_permission_list_show_frame, text=name)
            name_label.grid(row=row, column=2)
            check_btn = Checkbutton(self.manage_permission_list_show_frame, text='Được cho phép', onvalue=1, offvalue=0,
                                    command=lambda index=row: self.get_btn(index))
            check_btn.grid(row=row, column=3)
            if permission:
                check_btn.select()
            btn = Button(self.manage_permission_list_show_frame, text=f'Xóa',
                         command=lambda index=row: self.delete_db(index))
            self.label_list.append(photo_label)
            self.label_list.append(name_label)
            self.label_list.append(check_btn)
            self.label_list.append(btn)
            btn.grid(row=row, column=4)

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
            if self.last_capture_face is not None:
                self.model_recog.save_data_user(self.last_capture_face, name)
                util.msg_box('Success!', 'User was registered successfully !')
        except Exception as e:
            print(e)
        IS_REGISTER = False
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
