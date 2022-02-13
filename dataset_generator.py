import cv2
from mtcnn import MTCNN
import os
import time
from tqdm import tqdm
from frame_scrapping import get_cam_frame
from faces_extractor import get_faces_from_frame

AVAILABLE_CAMS = ['01', '03', '04', '06', '07', '08', '10', '11']
LABELS = ['arthur', 'barbara', 'brunna', 'douglas', 'eliezer', 'eslo', 'gustavo', 'jade', 
        'jessi', 'lais', 'lari', 'linna', 'lucas', 'maria', 'naty', 'paulo', 'scooby', 'tiago', 'vini']

SLEEPING_TIME = 10
IMG_SIZE = (160, 160)

DIR_PATH = 'dataset2/train/'
try:
    os.makedirs(DIR_PATH)
except FileExistsError:
    pass
finally:
    DIR_COUNT = len(os.listdir(DIR_PATH))

def choose_label():
    """
    Returns label according by user input
    """
    count = 0;
    
    print('\nWhose face is this?\n')
    for label in LABELS:
        print(str(count) + ' - '+ label + '\n')
        count += 1

    selected = input('Select a label by number or just leave blank to skip: ')
    if (selected == ''):
        raise Exception('Blank input')

    return LABELS[int(selected)];

def close_windows():
    """
    Kludge to make destroyAllWindows method work
    """
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

labelled_amount = 0
detector = MTCNN(min_face_size = 30)
cv2.startWindowThread()

while True:

    for cam in AVAILABLE_CAMS:

        frame = get_cam_frame(cam)
        cv2.imshow('Cam ' + cam,frame)
        cv2.waitKey(1000)

        faces = get_faces_from_frame(frame, detector, IMG_SIZE)

        for face in faces:
            cv2.imshow('label this face', face)
            cv2.waitKey(100)

            try:
                # get label input
                label = choose_label() 

                # verify label dir
                path = DIR_PATH + label
                if not os.path.exists(path): os.makedirs(path)

                # composes filename and saves labeled face
                filename = label + str(len(os.listdir(path)) + 1) + '.jpg'
                cv2.imwrite(path + '/' + filename, face)

                print('\nA face was labelled: ' + filename)
                labelled_amount += 1
            except:
                print('\nThe face was skipped')
            finally:
                close_windows()

        close_windows()    
        print('\n' + str(len(faces)) + ' faces were found on Cam ' + cam + ', and ' + str(labelled_amount) + ' faces were labeled in this session')

    print('\nSleeping for ' + str(SLEEPING_TIME) + ' seconds...\n')
    
    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)