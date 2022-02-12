import cv2

def extract_faces(frame, detector, min_confidence = 0.9, marker = True, padding = 10):
    """
    Returns extracted faces array.

    :param frame: array
    :param detector: MTCNN()
    :param min_confidence: float (defaults 0.9)
    :param marker: boolean (defaults True)
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
                extracted_faces.append(cropped_face)
                if marker: cv2.rectangle(frame, (x1-1,y1-1), (x2,y2), (100, 255,100), 1)

    return extracted_faces