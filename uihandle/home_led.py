import urllib.request

import numpy as np

import util
from model_detection import ModelDetection

model_detection = ModelDetection()
print('Model has been loaded')


def run_detect():
    IS_HAVING_PERSON = False
    DETECT_DATA = None
    while 1:
        img_resp = urllib.request.urlopen(util.url)
        img = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        DETECT_DATA, IS_HAVING_PERSON = model_detection.predict(img)
        if IS_HAVING_PERSON and DETECT_DATA is not None:
            util.set_led_state(util.LED_ON)
        else:
            util.set_led_state(util.LED_OFF)


if __name__ == "__main__":
    run_detect()
