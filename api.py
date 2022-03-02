from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND
from config import IMG_SIZE, FACENET_MODEL_KEY
from face_predictor import FacePredictor
from cam_scraper import CamScraper
import datetime
import asyncio

# seconds between scrape and predict cams
SLEEPING_TIME = 15

# cam frame visualization?
VISUALIZATION_ENABLED = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # list to store scraped cams with predicts
    app.state.cams = []
    
    # create face predictor and get labels
    app.state.predictor = FacePredictor(IMG_SIZE, FACENET_MODEL_KEY)
    app.state.labels = app.state.predictor.get_labels()
    
    # dict to store cams by label
    app.state.cams_by_labels = {}
    
    # initialize a dict for each label
    for label in app.state.labels:
        app.state.cams_by_labels[label] = {}

    # create cam scrapper and launch browser
    app.state.cam_scraper = CamScraper()
    await app.state.cam_scraper.launch_browser()

    # create task to update cams with predicted faces
    asyncio.create_task(scrape_and_predict_cams())

async def scrape_and_predict_cams():
    while True:
        try:
            for label in app.state.labels:        
                # copy prev cams
                if 'cams' in app.state.cams_by_labels[label]:
                    prev_cams = app.state.cams_by_labels[label]['cams']
                    if len(prev_cams) > 0:
                        app.state.cams_by_labels[label]['prev_cams'] = prev_cams

                # reset cams
                app.state.cams_by_labels[label]['cams'] = []

            # scrap available cams
            cams = await app.state.cam_scraper.scrape_available_cams()
            scrape_timestamp = datetime.datetime.now()
            
            for cam in cams:
                # get cam frame by snapshot link
                frame = app.state.cam_scraper.scrape_cam_frame(cam['snapshot_link'])
                frame_timestamp = datetime.datetime.now()

                # get recognized faces on cam frame
                recognized_faces = app.state.predictor.predict_frame(frame, visualization_enabled = VISUALIZATION_ENABLED)

                # append recognized faces on cam dict
                cam['recognized_faces'] = recognized_faces

                # store timestamps
                cam['scrape_timestamp'] = scrape_timestamp
                cam['frame_timestamp'] = frame_timestamp

                # update cams by labels list
                for face in recognized_faces:
                    app.state.cams_by_labels[face['label']]['cams'].append(cam)

            # update cams list
            app.state.cams = cams

        except Exception as e:
            print(str(e))
        
        await asyncio.sleep(SLEEPING_TIME)

@app.get("/")
def cams_redirect():
    return RedirectResponse(url='/cams')

@app.get("/cams")
def get_cams():
    return app.state.cams if len(app.state.cams) > 0 else Response(status_code=HTTP_204_NO_CONTENT)

@app.get("/labels")
def get_labels():
    return app.state.labels

@app.get("/cams/labels/{label}")
def get_cams_by_label(label):
    return app.state.cams_by_labels[label] if label in app.state.labels else Response(status_code=HTTP_404_NOT_FOUND)

@app.get("/cams/labels")
def get_cams_by_labels():
    return app.state.cams_by_labels

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.cam_scraper.close_browser()