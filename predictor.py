from config import IMG_SIZE, DETECTOR_MIN_FACE_SIZE, FACENET_MODEL_KEY, MODEL_JOBLIB_PATH, LABEL_ENCODER_JOBLIB_PATH
from cam_scraper import get_available_cams, get_cam_frame
from face_extractor import get_faces_from_frame
from utils import close_windows, draw_label
from keras_facenet import FaceNet
from mtcnn import MTCNN
from joblib import load
import numpy as np
import json
import cv2
import sys

# "-v" as first command line argument to enable visualization mode 
VISUALIZATION_ENABLED = True if len(sys.argv) > 1 and sys.argv[1] == '-v' else False

# integer as second command line argument to set visualization time (miliseconds) per cam
CAM_VISUALIZATION_TIME = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 2000

# get ML models
detector = MTCNN(min_face_size = DETECTOR_MIN_FACE_SIZE)
embedder = FaceNet(key = FACENET_MODEL_KEY)
classifier = load(MODEL_JOBLIB_PATH)

# restore label encoder
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
    y_label = classifier.predict(embedded_face)
    y_prob = classifier.predict_proba(embedded_face)

    # label decoding
    label = label_encoder.inverse_transform(y_label)

    return label, y_prob

# scrap available cams
available_cams = get_available_cams()

# iterate over cams
for cam in available_cams:

    # get cam frame by snapshot link
    frame = get_cam_frame(cam['snapshot_link'])

    # get faces on frame
    faces = get_faces_from_frame(frame, detector, IMG_SIZE, marker = VISUALIZATION_ENABLED)

    print(str(len(faces)) + ' faces were found on ' + cam['name'] + ' - ' + cam['location'] + '...')

    # create list to store recognized_faces per cam
    cam['recognized_faces'] = []
    
    # predict all faces
    for face in faces:
        label, y_prob = predict(face['pixels'])
        
        # get probability percentage
        probability = np.amax(y_prob[0]) * 100

        # append recognized face on cam recognized faces list
        cam['recognized_faces'].append({"name": label[0], "probability": probability, "coordinates": face['coordinates']})

        # draw label text
        if VISUALIZATION_ENABLED: draw_label(frame, face['coordinates'], label, probability)
    
    # cam frame visualization 
    if VISUALIZATION_ENABLED:
        cv2.imshow(cam['name'] + ' - ' + cam['location'], frame)
        cv2.waitKey(CAM_VISUALIZATION_TIME)
        close_windows()

print(json.dumps(available_cams, sort_keys=False, indent=4))