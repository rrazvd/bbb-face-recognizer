import cv2
import os
import time
from tqdm import tqdm
from mtcnn import MTCNN
from frame_scrapping import get_cam_frame

DIR_PATH = 'dataset'
try:
    os.mkdir(DIR_PATH)
except FileExistsError:
    pass
finally:
    DIR_COUNT = len(os.listdir(DIR_PATH))

SLEEPING_TIME = 15
AVAILABLE_CAMS = ['01', '03', '04', '06', '07', '08', '10', '11']
IMG_SIZE = (160, 160)

def get_cropped_face(box, frame, padding = 10):
    x, y, w, h = box
    x1, y1 = (x - padding), (y - padding)
    x2, y2 = (x + w + padding), (y + h + padding)
    cv2.rectangle(frame, (x1-1,y1-1), (x2,y2), (100, 255,100), 1)
    return frame[y1:y2, x1:x2]

detector = MTCNN(min_face_size = 30)
count = DIR_COUNT
cv2.startWindowThread()

while True:
    for cam in AVAILABLE_CAMS:
        frame = get_cam_frame(cam)
        faces = detector.detect_faces(frame)
    
        for face in faces:      
            if face['confidence'] >= 0.9: 
                cropped_face = get_cropped_face(face['box'], frame)
                if len(cropped_face) > 0: 
                    resized_img = cv2.resize(cropped_face, IMG_SIZE)
                    cv2.imwrite(DIR_PATH + "/face"+str(count)+".jpg", resized_img)
                    count += 1

        cv2.imshow('image',frame)
        cv2.waitKey(2000)

        print('\n' + str(len(faces)) + ' faces were found on Cam ' + cam)

    print('\nSleeping for '+ str(SLEEPING_TIME) +' seconds...\n')
    
    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)

    cv2.destroyAllWindows()

