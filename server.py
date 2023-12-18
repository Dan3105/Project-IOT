import asyncio
import urllib.request
import cv2
import numpy as np
import requests
import io
import time
from model_detection import ModelDetection, decode_image, encode_image

url = 'http://192.168.0.119/cam-lo.jpg'

# Set your bot token and chat ID
bot_token = "6741379537:AAErR_8MoBNXsoOpC0bDmkFlEn5EaYJFn6A"
chat_id = "6693077264"

connected_clients = dict()
model_detection = ModelDetection()
print('Model has been loaded')

def send_notification(data):
    print('Data detect human')
    # Send an image
    image_file = io.BytesIO(data)   # Convert the received binary data to a file-like object
    files = {'photo': ('image.jpg', image_file)}    # Prepare the files parameter
    image_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto?chat_id={chat_id}"
    requests.post(image_url, files=files)

    # Send a text message
    text_message = "Detect human!"
    text_url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text_message}"
    requests.get(text_url)

    # Wait for 5 seconds before continuing to detect
    #asyncio.sleep(5)

def run_detect():
    while(1):
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        
        detect_data, _ishavingperson = model_detection.predict(imgnp)
        if _ishavingperson:
            send_notification(detect_data)
            time.sleep(5)


if __name__ == "__main__":
    run_detect()
