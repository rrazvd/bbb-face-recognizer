from config import IMG_SIZE, FACENET_MODEL_KEY
from face_predictor import FacePredictor
from cam_scraper import CamScraper
from fastapi import FastAPI, Response
from starlette.responses import RedirectResponse
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND
import asyncio

# seconds between scrape and predict cams
SLEEPING_TIME = 30

# cam frame visualization?
VISUALIZATION_ENABLED = False

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    
    # list to store scraped cams with predicts
    app.state.cams = []
    
    # creste face predictor and get labels
    app.state.predictor = FacePredictor(IMG_SIZE, FACENET_MODEL_KEY)
    app.state.labels = app.state.predictor.get_labels()
    
    # dict to store cams by label
    app.state.cams_by_label = {}
    for label in app.state.labels:
        app.state.cams_by_label[label] = []

    # create cam scrapper and launch browser
    app.state.cam_scraper = CamScraper()
    await app.state.cam_scraper.launch_browser()

    # create task to update cams with predicted faces
    asyncio.create_task(scrape_and_predict_cams())

async def scrape_and_predict_cams():
   
    while True:
        try:
            # scrap available cams
            cams = await app.state.cam_scraper.scrape_available_cams()
            
            for cam in cams:
                # get cam frame by snapshot link
                frame = app.state.cam_scraper.scrape_cam_frame(cam['snapshot_link'])

                # get recognized faces on cam frame
                recognized_faces = app.state.predictor.predict_frame(frame, visualization_enabled = VISUALIZATION_ENABLED)

                # append recognized faces on cam dict
                cam['recognized_faces'] = recognized_faces

                # update cams by label dict
                for face in recognized_faces:
                    app.state.cams_by_label[face['label']].append(cam)

            # update cams dict
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

@app.get("/cams/{label}")
def get_cams_by_label(label):
    return app.state.cams_by_label[label] if label in app.state.labels else Response(status_code=HTTP_404_NOT_FOUND)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.cam_scraper.close_browser()