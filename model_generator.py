import cv2
import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import dump
from face_extractor import get_faces_from_dir

TRAIN_DATASET_PATH = 'dataset2/train'
VAL_DATASET_PATH = 'dataset2/val'

def load_dataset(path):
    """
    Returns labeled dataset.

    :param path: path string to dir of dataset

    :return X, y nparrays
    """
    X, y = [], []

    for label in os.listdir(path):
        faces = get_faces_from_dir(path + '/' + label + '/')
        labels = [label for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)

    return np.asarray(X), np.asarray(y)


# load train and val datasets
X_train, y_train = load_dataset(TRAIN_DATASET_PATH)
X_val, y_val = load_dataset(VAL_DATASET_PATH)

# face embedding
embedder = FaceNet(key='20170511-185253')
X_train = embedder.embeddings(X_train)
X_val = embedder.embeddings(X_val)

# label encoding
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_val = label_encoder.transform(y_val)

# fit svm classifier model
model = SVC(kernel='linear', probability = True)
model.fit(X_train, y_train)

# predict
y_train_p = model.predict(X_train)
y_val_p = model.predict(X_val)

# metrics
acc_score = accuracy_score(y_val, y_val_p)
matrix = confusion_matrix(y_val, y_val_p)

print('Accuracy: ' + str(acc_score*100))
print('Confusion matrix: \n' + str(matrix))

# save svm model and label encoder
dump(model, 'model.joblib')
dump(label_encoder, 'label_encoder.joblib')