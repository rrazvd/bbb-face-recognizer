from config import TRAIN_DATASET_DIR_PATH, VAL_DATASET_DIR_PATH, IMG_SIZE
from cam_scraper import CamScraper
from face_extractor import FaceExtractor
from utils import close_windows
from tqdm import tqdm
import asyncio
import cv2
import sys
import os

LABELS = ['arthur', 'douglas', 'eliezer', 'eslo', 'gustavo', 'jade', 'jessi', 'lais', 'linna', 'lucas', 'naty', 'paulo', 'scooby', 'vini']

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


async def main():
   
    # get face extractor
    face_extractor = FaceExtractor(IMG_SIZE)

    # labeled amount per session counter
    labeled_amount = 0

    # scrape available cams
    cs = CamScraper()
    await cs.launch_browser()
    AVAILABLE_CAMS = await cs.scrape_available_cams()
    await cs.close_browser()

    cv2.startWindowThread()

    while True:
        
        # iterate over cams
        for cam in AVAILABLE_CAMS:

            # get cam frame by snapshot link and show it
            frame = cs.scrape_cam_frame(cam['snapshot_link'])

            # get faces on frame
            faces = face_extractor.get_faces_from_frame(frame, IMG_SIZE)

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
            await asyncio.sleep(1)

asyncio.run(main())