import cv2
import numpy as np
import requests

def get_cam_frame(cam_code):
    """
    Returns frame array according by cam code.

    :param cam_code: string

    :return frame array
    """
    
    resp = requests.get('https://live-thumbs.video.globo.com/bbb'+cam_code+'/snapshot/', stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)