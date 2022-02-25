
from predictor import Predictor
from cam_scraper import CamScraper
from fastapi import FastAPI, Response
from starlette.responses import RedirectResponse
from starlette.status import HTTP_204_NO_CONTENT
import asyncio

# seconds between scrape and predict cams
SLEEPING_TIME = 30

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    app.state.cams = []
    app.state.predictor = Predictor()

    app.state.cam_scraper = CamScraper()
    await app.state.cam_scraper.launch_browser()

    # create task to update cams with recognized faces
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
                recognized_faces = app.state.predictor.predict_frame(frame, visualization_enabled = False)

                # append recognized faces on cam dict
                cam['recognized_faces'] = recognized_faces

            # update cams state
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
    return app.state.predictor.get_labels()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.cam_scraper.close_browser()