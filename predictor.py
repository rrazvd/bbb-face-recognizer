import time
from joblib import load
from mtcnn import MTCNN
from tqdm import tqdm
from cam_scrapper import get_available_cams, get_cam_frame
from face_extractor import get_faces_from_frame
from keras_facenet import FaceNet
from utils import close_windows, draw_label
import numpy as np
import cv2

SLEEPING_TIME = 10
IMG_SIZE = (160, 160)

detector = MTCNN(min_face_size = 30)
embedder = FaceNet(key='20170511-185253')

model = load('model.joblib')
label_encoder = load('label_encoder.joblib')

def predict(face):
    """
    Predict a face and returns label and probability array.

    :param face: face pixel array

    :return label and probability array
    """
    # embedding
    embedder = FaceNet(key='20170511-185253')
    embedded_face = embedder.embeddings([face])

    # predict
    y_label = model.predict(embedded_face)
    y_prob = model.predict_proba(embedded_face)

    # label decoding
    label = label_encoder.inverse_transform(y_label)

    return label, y_prob

cv2.startWindowThread()

while True:

    for cam in get_available_cams():

        # get cam frame by cam code
        frame = get_cam_frame(cam)

        # get faces on frame
        faces = get_faces_from_frame(frame, detector, IMG_SIZE, marker=True)

        print(str(len(faces)) + ' faces were found on Cam ' + cam + '...')

        # predict all faces
        for face in faces:
            label, y_prob = predict(face['pixels'])
            probability = np.amax(y_prob[0]) * 100
            draw_label(frame, face['coordinates'], label, probability)
        
        cv2.imshow('Cam ' + cam,frame)
        cv2.waitKey(2000)

    close_windows()

    print('\nSleeping for ' + str(SLEEPING_TIME) + ' seconds...\n')
    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)