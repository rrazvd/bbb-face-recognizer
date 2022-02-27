from mtcnn import MTCNN
import cv2

class FaceExtractor():
    """
    Class that represents a face extractor.
    """
    def __init__(self, output_size, min_face_size = 30, min_confidence = 0.9):
        """
        :param output_size: int tuple to define in pixels the output size
        :param min_confidence: float (defaults 0.9)
        :param min_face_size: minimum size of the face to detect
        """
        self.output_size = output_size
        self.detector = MTCNN(min_face_size = min_face_size)
        self.min_confidence = min_confidence

    def get_faces_from_frame(self, frame, face_marker = False, padding = 10):
        """
        Returns extracted faces array from frame.

        :param frame: frame pixel array
        :param face_marker: boolean (defaults False)
        :param padding: int (defaults 10)

        :return extracted faces arrays
        """
        extracted_faces = []
        faces = self.detector.detect_faces(frame)

        for face in faces:      

            # filters faces according minimum confidence
            if face['confidence'] >= self.min_confidence: 
                
                x, y, w, h = face['box']

                # defines box coordinates with padding
                x1, y1 = (x - padding), (y - padding)
                x2, y2 = (x + w + padding), (y + h + padding)

                # crops face from frame using box coordinates
                cropped_face = frame[y1:y2, x1:x2]

                if len(cropped_face) > 0: 

                    # resize according by output size and append on the list
                    resized_face = cv2.resize(cropped_face, self.output_size)
                    extracted_faces.append({"pixels": resized_face, "coordinates": {"topLeft": (x1, y1), "bottomRight": (x2,y2)}})

                    # renders a green face marker if needed
                    if face_marker: cv2.rectangle(frame, (x1-1,y1-1), (x2,y2), (100, 255,100), 1)

        return extracted_faces