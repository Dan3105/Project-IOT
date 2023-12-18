from ultralytics import YOLO
import numpy as np
import cv2
import io
import base64
from PIL import Image
def decode_image(byte_array):
    image_data = np.array(Image.open(io.BytesIO(byte_array)))
    return image_data

def encode_image(image):
    _, JPEG = cv2.imencode('.jpeg', image)
    return JPEG.tobytes()

class ModelDetection:
    def __init__(self):
        self.model = YOLO('./Model/yolov8n.pt')
        self.model.fuse()
    def predict(self, byte_array):
        """
        return image_with_detect_box, boolean which return is have human in this image
        """
        decode = decode_image(byte_array=byte_array)
        results = self.model.predict(decode, verbose=False, imgsz=320, classes=[0])
        is_having_person = len(results[0].boxes) > 0 
        return encode_image(results[0].plot()), is_having_person