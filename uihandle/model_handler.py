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

class ModelRecognition:
    def __init__(self, db_path, db_img, threshold=0.7):
        self._model = InceptionResnetV1(pretrained='vggface2').eval()
        #self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self._model.to(self._device)
        self._threshold = threshold
        self._db_path = db_path
        self._db_img = db_img

    def __distance(self, encode_input, encode_user):
        return np.linalg.norm(encode_input - encode_user)
    def predict(self, image):
        """
        image: face_human, result which model above return
        return:
            id, name: person
        """
        convert_to_tensor = torch.tensor(image)
        convert_to_tensor = convert_to_tensor.permute(2, 0, 1)

        normalized_tensor = convert_to_tensor.divide(255)
        encode_input = self._model(normalized_tensor.unsqueeze(0))
        
        users = pd.read_csv(self._db_path)
        for index, row in users.iterrows():
            image_path = os.path.join(self._db_img, row['Image'])
            image_user = cv2.imread(image_path)


            user_to_tensor = torch.tensor(image_user)
            user_to_tensor = user_to_tensor.permute(2, 0, 1)
            normalized_user_tensor = user_to_tensor.divide(255)

            encode_user = self._model(normalized_user_tensor.unsqueeze(0))

            np_encode_input = encode_input.detach().numpy()
            np_encode_user = encode_user.detach().numpy()

            dist = self.__distance(np_encode_input, np_encode_user)
            if dist < self._threshold:
                return row['Name']
            #print(dist)
            else:
                print(f'Image of {row["Name"]} is not correct')
        return None
    
    def save_data_user(self, image, name):
        name_format = name.split(' ')[-1]+'.png'
        name_path = os.path.join(self._db_img, name_format)
        cv2.imwrite(name_path, image)
        new_row_data = {'Name': name, 'Image': name_format}
        df = pd.read_csv(self._db_path)
        df= pd.concat([df, pd.DataFrame([new_row_data])], ignore_index=True)
        df.to_csv(self._db_path, index=False)