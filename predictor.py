from config import IMG_SIZE, DETECTOR_MIN_FACE_SIZE, FACENET_MODEL_KEY, MODEL_JOBLIB_PATH, LABEL_ENCODER_JOBLIB_PATH
from cam_scrapper import get_available_cams, get_cam_frame
from face_extractor import get_faces_from_frame
from utils import close_windows, draw_label
from keras_facenet import FaceNet
from mtcnn import MTCNN
from joblib import load
from tqdm import tqdm
import numpy as np
import time
import cv2

SLEEPING_TIME = 10

detector = MTCNN(min_face_size = DETECTOR_MIN_FACE_SIZE)
embedder = FaceNet(key = FACENET_MODEL_KEY)
model = load(MODEL_JOBLIB_PATH)
label_encoder = load(LABEL_ENCODER_JOBLIB_PATH)

def predict(face):
    """
    Predict a face and returns label and probability array.

    :param face: face pixel array

    :return label and probability array
    """
    # embedding
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