import cv2
import numpy as np

def get_faces_from_frame(frame, detector, output_size, min_confidence = 0.9, marker = False, padding = 10):
    """
    Returns extracted faces array from frame.

    :param frame: array
    :param detector: MTCNN()
    :param output_size: int tuple
    :param min_confidence: float (defaults 0.9)
    :param marker: boolean (defaults False)
    :param padding: int (defaults 10)

    :return extracted faces arrays
    """
    
    extracted_faces = []
    faces = detector.detect_faces(frame)

    for face in faces:      

        if face['confidence'] >= min_confidence: 
            x, y, w, h = face['box']
            x1, y1 = (x - padding), (y - padding)
            x2, y2 = (x + w + padding), (y + h + padding)

            cropped_face = frame[y1:y2, x1:x2]

            if len(cropped_face) > 0: 
                resized_face = cv2.resize(cropped_face, output_size)
                extracted_faces.append(resized_face)
                if marker: cv2.rectangle(frame, (x1-1,y1-1), (x2,y2), (100, 255,100), 1)

    return extracted_faces