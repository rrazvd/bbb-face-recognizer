from config import TRAIN_DATASET_DIR_PATH, VAL_DATASET_DIR_PATH, IMG_SIZE, DETECTOR_MIN_FACE_SIZE
from cam_scraper import get_cam_frame, get_available_cams
from face_extractor import get_faces_from_frame
from utils import close_windows
from mtcnn import MTCNN
from tqdm import tqdm
import time
import cv2
import sys
import os

LABELS = ['arthur', 'barbara', 'brunna', 'douglas', 'eliezer', 'eslo', 'gustavo', 'jade', 
        'jessi', 'lais', 'lari', 'linna', 'lucas', 'maria', 'naty', 'paulo', 'scooby', 'tiago', 'vini']

# time between cams iteration
SLEEPING_TIME = 5

# "-val" as first command line argument if want populate validation dataset (defaults to train)
DIR_PATH = VAL_DATASET_DIR_PATH if len(sys.argv) > 1 and sys.argv[1] == '-val' else TRAIN_DATASET_DIR_PATH
if not os.path.exists(DIR_PATH): os.makedirs(DIR_PATH)

print('\nPopulating train dataset...\n') if DIR_PATH == TRAIN_DATASET_DIR_PATH else print('\nPopulating validation dataset...\n')

# create dir for each label
for label in LABELS:
    # create in train dir
    path = TRAIN_DATASET_DIR_PATH + label
    if not os.path.exists(path): os.makedirs(path)
    # create in validation dir
    path = VAL_DATASET_DIR_PATH + label
    if not os.path.exists(path): os.makedirs(path)

def choose_label():
    """
    Returns label according by user input
    """
    count = 0;

    print('\nWhose face is this?\n')
    for label in LABELS:
        print(str(count) + ' - ' + label)
        count += 1

    selected = input('\nSelect a label by number or just leave blank to skip: ')
    if (selected == ''):
        raise Exception('Blank input')

    return LABELS[int(selected)];

# get detector model
detector = MTCNN(min_face_size = DETECTOR_MIN_FACE_SIZE)

# labeled amount per session counter
labeled_amount = 0

# scrap available cams
AVAILABLE_CAMS = get_available_cams()

cv2.startWindowThread()

while True:

    # iterate over cams
    for cam in AVAILABLE_CAMS:

        # get cam frame by snapshot link and show it
        frame = get_cam_frame(cam['snapshot_link'])

        # get faces on frame
        faces = get_faces_from_frame(frame, detector, IMG_SIZE)

        for face in faces:

            # show the recognized face
            cv2.imshow('label this face', face['pixels'])
            cv2.waitKey(100)

            try:
                # get label input
                label = choose_label() 

                # verify label dir
                path = DIR_PATH + label
                if not os.path.exists(path): os.makedirs(path)

                # composes filename and saves labeled face
                filename = label + str(len(os.listdir(path)) + 1) + '.jpg'
                cv2.imwrite(path + '/' + filename, face['pixels'])

                print('\nA face was labeled: ' + filename)
                labeled_amount += 1
            except:
                print('\nThe face was skipped')
            finally:
                close_windows()

        close_windows()
        print('\n' + str(len(faces)) + ' faces were found on ' + cam['name'] + ' - ' + cam['location'] + ' and ' + str(labeled_amount) + ' faces were labeled in this session')

    print('\nSleeping for ' + str(SLEEPING_TIME) + ' seconds...\n')

    for i in tqdm(range(SLEEPING_TIME)):
        time.sleep(1)