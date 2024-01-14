from ultralytics import YOLO
import cv2
import math
import cvzone

from facenet_pytorch import InceptionResnetV1
import torch
import os
import pandas as pd
import numpy as np

class ModelAntiSpoffing:
    def __init__(self, asmodelpath, confidence = 0.8):
        self._model = YOLO(asmodelpath)
        self._confidence = confidence
        self._classNames = ['device', 'live', 'mask', 'photo']

    def detect(self, image):
        """
        image: image in format BGR
        return:
            + image ORIGINAL with green box have the HIGHEST conf label is 'live'
            + image above but only contain face
        """

        results = self._model(image, stream=True, verbose=False)
        face_human, highest_conf = None, 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])

                if conf > self._confidence:
                    if self._classNames[cls] == 'live':
                        if conf > highest_conf:
                            highest_conf = conf
        
                            color = (0, 255, 0)
                            # Bounding Box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w, h = x2 - x1, y2 - y1

                            face_human = image[y1:y1+h, x1:x1+w].copy()

                            cvzone.cornerRect(image, (x1, y1, w, h),colorC=color,colorR=color)
                            cvzone.putTextRect(image, f'{self._classNames[cls].upper()} {int(conf*100)}%',
                                    (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=color,
                                    colorB=color)
                            
        return image, face_human

class ModelDetectorFace:
    def __init__(self, model_path):
        self._model = cv2.FaceDetectorYN_create(model_path, "", (0,0))
        self._model.setScoreThreshold(0.87)
        self.__scaling_size = 320
    def __format_image(self, image):
        format_image = cv2.resize(image, (0,0), fx=self.__scaling_size/image.shape[0], fy=self.__scaling_size/image.shape[0])
        return format_image
    def get_encode_face(self, image):
        """
        image: numpy array format BGR
        return array result
        """
        format_image = self.__format_image(image)
        height, width, _ = format_image.shape
        self._model.setInputSize((width, height))
        try:
            _, encode = self._model.detect(format_image)
            return encode
        except:
            return None

class ModelRecognition:
    def __init__(self, model_detect_path, model_recog_path, db_path, db_img, threshold=0.1):
        self._model_recog = cv2.FaceRecognizerSF_create(model_recog_path, "")
        #self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self._model.to(self._device)
        self._model_detect = ModelDetectorFace(model_detect_path)
        self._threshold = threshold
        self._db_path = db_path
        self._db_img = db_img

    def __distance(self, encode_input, encode_user):
        score = self._model_recog.match(encode_input, encode_user, cv2.FaceRecognizerSF_FR_NORM_L2)
        return score
    def predict(self, image):
        """
        image: face_human, result which model above return
        return:
            id, name: person
        """
        input_encode = self._model_detect.get_encode_face(image)
        if input_encode is None:
            return None
        
        users = pd.read_csv(self._db_path)
        smallest_score_person = 100
        matches_person = None
        for index, row in users.iterrows():
            image_path = os.path.join(self._db_img, row['Image'])
            image_user = cv2.imread(image_path)

            encode_user = self._model_detect.get_encode_face(image_user)
            if encode_user is not None:
                score = self.__distance(input_encode, encode_user)
                #print(score)
                if score < smallest_score_person and score < self._threshold:
                    smallest_score_person = score 
                    matches_person = row
            else:
                print(f'Image of {row["Name"]} is not correct')
        
        if matches_person["Permission" == 0]:
            return None
        return matches_person["Name"]
    
    def save_data_user(self, image, name):
        if image is None:
            print('Image is unreal')
            return
        name_format = name.split(' ')[-1]+'.png'
        name_path = os.path.join(self._db_img, name_format)
        #name_path = name_format
        cv2.imwrite(name_path, image)
        print(name_path)
        new_row_data = {'Name': name, 'Image': name_format, 'Permission' : 1}
        df = pd.read_csv(self._db_path)
        df= pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
        df.to_csv(self._db_path, index=False)