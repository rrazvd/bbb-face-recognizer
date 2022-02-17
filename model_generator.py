from config import TRAIN_DATASET_DIR_PATH, VAL_DATASET_DIR_PATH, MODEL_FILES_DIR_PATH, MODEL_JOBLIB_PATH, LABEL_ENCODER_JOBLIB_PATH, FACENET_MODEL_KEY
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from face_extractor import get_faces_from_dir
from keras_facenet import FaceNet
from sklearn.svm import SVC
from joblib import dump
import numpy as np
import os

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
X_train, y_train = load_dataset(TRAIN_DATASET_DIR_PATH)
X_val, y_val = load_dataset(VAL_DATASET_DIR_PATH)

# face embedding
embedder = FaceNet(key = FACENET_MODEL_KEY)
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
if not os.path.exists(MODEL_FILES_DIR_PATH): os.makedirs(MODEL_FILES_DIR_PATH)
dump(model, MODEL_JOBLIB_PATH)
dump(label_encoder, LABEL_ENCODER_JOBLIB_PATH)