import cv2, os

def get_faces_from_frame(frame, detector, output_size, min_confidence = 0.9, marker = False, padding = 10):
    """
    Returns extracted faces array from frame.

    :param frame: frame pixel array
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

        # filters faces according minimum confidence
        if face['confidence'] >= min_confidence: 
            
            x, y, w, h = face['box']

            # defines box coordinates with padding
            x1, y1 = (x - padding), (y - padding)
            x2, y2 = (x + w + padding), (y + h + padding)

            # crops face from frame using box coordinates
            cropped_face = frame[y1:y2, x1:x2]

            if len(cropped_face) > 0: 

                # resize according by output size and append on the list
                resized_face = cv2.resize(cropped_face, output_size)
                extracted_faces.append({"pixels": resized_face, "coordinates": ((x1, y1), (x2,y2))})

                # renders a green face marker if needed
                if marker: cv2.rectangle(frame, (x1-1,y1-1), (x2,y2), (100, 255,100), 1)

    return extracted_faces

def get_faces_from_dir(path):
    """
    Returns array with available faces on dir.

    :param path: path string to dir

    :return array of faces
    """
    faces = []
    for filename in os.listdir(path):
        face = cv2.imread(path+'/'+filename)
        faces.append(face)
    return faces