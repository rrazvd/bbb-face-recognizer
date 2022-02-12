import cv2
from mtcnn import MTCNN
import os
import time
from tqdm import tqdm
from frame_scrapping import get_cam_frame
from faces_extractor import extract_faces

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

count = DIR_COUNT
detector = MTCNN(min_face_size = 30)
cv2.startWindowThread()

while True:
    for cam in AVAILABLE_CAMS:
        frame = get_cam_frame(cam)
        faces = extract_faces(frame, detector)

        for face in faces:
            resized_img = cv2.resize(face, IMG_SIZE)
            cv2.imwrite(DIR_PATH + "/face"+str(count)+".jpg", resized_img)
            count += 1

        cv2.imshow('image',frame)
        cv2.waitKey(2000)

        print('\n' + str(len(faces)) + ' faces were found on Cam ' + cam)

    print('\nSleeping for '+ str(SLEEPING_TIME) +' seconds...\n')
    
    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)

    cv2.destroyAllWindows()

