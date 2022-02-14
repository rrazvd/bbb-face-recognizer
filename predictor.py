import time
from joblib import load
from mtcnn import MTCNN
from tqdm import tqdm
from cam_scrapper import get_available_cams, get_cam_frame
from face_extractor import get_faces_from_frame
from keras_facenet import FaceNet
from matplotlib import pyplot
import numpy as np
import cv2

SLEEPING_TIME = 10
IMG_SIZE = (160, 160)

detector = MTCNN(min_face_size = 30)
embedder = FaceNet(key='20170511-185253')

model = load('model.joblib')
label_encoder = load('label_encoder.joblib')

def plot_predicted_face(face, predicted, probability_arr):
    """
    Plot predicted face with label and probability.

    :param face: face pixel array
    :param predicted: predicted label string
    :param probability_arr: probability array
    """
    pyplot.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    title = '%s (%.2f)' % (predicted[0], np.amax(probability_arr[0]) * 100)  
    pyplot.title(title)
    pyplot.show()

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

while True:

    for cam in get_available_cams():

        # get cam frame by cam code
        frame = get_cam_frame(cam)

        # get faces on frame
        faces = get_faces_from_frame(frame, detector, IMG_SIZE)

        print(str(len(faces)) + ' faces were found on Cam ' + cam + '...')

        # predict all faces
        for face in faces:
            label, y_prob = predict(face)
            plot_predicted_face(face, label, y_prob)
        
        time.sleep(1)

    print('\nSleeping for ' + str(SLEEPING_TIME) + ' seconds...\n')
    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)