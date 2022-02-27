from config import MODEL_JOBLIB_PATH, LABEL_ENCODER_JOBLIB_PATH
from keras_facenet import FaceNet
from face_extractor import FaceExtractor
from utils import close_windows
from joblib import load
import numpy as np
import cv2

class FacePredictor:
    """
    Class that represents a face predictor.
    """
    def __init__(self, img_size, facenet_key):
        """
        Predict a face and returns label and probability array.
        :param img_size: output size of image
        :param facenet_key: string of facenet model key
        """
        self.extractor = FaceExtractor(img_size)
        self.embedder = FaceNet(key = facenet_key)
        self.classifier = load(MODEL_JOBLIB_PATH)
        self.label_encoder = load(LABEL_ENCODER_JOBLIB_PATH)

    def get_labels(self):
        """
        Returns label list used by model classifier

        :return label list
        """
        return self.label_encoder.classes_.tolist()

    def predict(self, face):
        """
        Predict a face and returns label and probability array.

        :param face: face pixel array

        :return label and probability array
        """
        # embedding
        embedded_face = self.embedder.embeddings([face])

        # predict
        y_label = self.classifier.predict(embedded_face)
        y_prob = self.classifier.predict_proba(embedded_face)

        # get probability percentage from predicted label
        label_prob = np.amax(y_prob[0]) * 100

        # label decoding
        label = self.label_encoder.inverse_transform(y_label)

        return label, label_prob, y_prob
    
    def predict_frame(self, frame, visualization_enabled = False, visualization_time = 2000):
        """
        Predict a frame and returns a list of recognized faces.

        :param frame: frame pixel array
        :param visualization_enabled: boolean
        :param visualization_time (miliseconds): int 

        :return list of recognized faces
        """
        
        # get faces on frame
        faces = self.extractor.get_faces_from_frame(frame, face_marker = visualization_enabled)

        # create list to store recognized_faces per frame
        recognized_faces = []
        
        # predict all faces
        for face in faces:
            label, label_prob, y_prob = self.predict(face['pixels'])
            recognized_faces.append({"label": label[0], "probability": label_prob, "coordinates": face['coordinates']})
            
            # draw label text
            if visualization_enabled: self.draw_label(frame, face['coordinates'], label, label_prob)
        
        # frame visualization 
        if visualization_enabled:
            cv2.imshow('frame visualization', frame)
            cv2.waitKey(visualization_time)
            close_windows()

        return recognized_faces

    
    def draw_label(self, frame, coordinates, label, probability):
        """
        Draws the text label with probability above box face.

        :param frame: frame pixel array
        :param coordinates: tuple with box face coordinates
        :param label: string of predicted label
        :param probability: float of probability
        """
        x1, y1 = coordinates['topLeft']
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.45
        fontColor = (0,255,0)

        text = '%s (%.2f)' % (label[0], probability) 
        cv2.putText(frame, text, (x1, y1 - 10), font, fontScale, fontColor, 1, 2) 