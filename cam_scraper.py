from pyppeteer import launch
import numpy as np
import requests
import asyncio
import json
import cv2

def get_available_cams():
    """
    Returns scraped available cams.

    :return array of cams dict
    """
    async def interceptResponse(response, target_url, result):
        if (response.url.startswith(target_url)):
            try:
                if "application/json" in response.headers.get("content-type", ""):
                    json_data = await response.json()
                    result.append(json_data['data']['localizedOffer']['paginatedItems']['resources'])
                    return
            except json.decoder.JSONDecodeError:
                pass
                
    async def browser_launcher():
        browser = await launch()
        page = await browser.newPage()

        result = []
        target_url = 'https://cloud-jarvis.globo.com/graphql?operationName=getOfferBroadcastByIdAndAffiliateCode'
        page.on('response', 
           lambda response: asyncio.ensure_future(interceptResponse(response, target_url, result)))

        await page.goto('https://globoplay.globo.com/categorias/big-brother-brasil/')
        await browser.close()
        return result

    print('\nScrapping cams...\n')
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(browser_launcher())
    loop.close()

    if (len(result) == 0):
        raise Exception('Unable to perform cam scraping')

    cams = []
    for cam in result[0]:
        cams.append(
            {'name': cam['name'], 'location': cam['media']['headline'], 'snapshot_link': cam['media']['liveThumbnail'], 
            'slug': cam['slug'], 'media_id': cam['mediaId'],
            'stream_link': 'https://globoplay.globo.com/'+cam['slug']+'/ao-vivo/'+cam['mediaId']+'/?category=bbb'}
        )

    print(str(len(cams)) + ' cams were scraped.\n')

    return cams

def get_cam_frame(snapshot_link):
    """
    Returns frame array according by snapshot link.

    :param snapshot_link: url string of snapshot link

    :return frame array
    """
    resp = requests.get(snapshot_link, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

