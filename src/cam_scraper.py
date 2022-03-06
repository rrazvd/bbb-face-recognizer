from pyppeteer import launch
import numpy as np
import requests
import asyncio
import cv2
import json

class CamScraper():
    """
    Class that represents a cam scraper.
    """
    def __init__(self):
        self.browser = None
        self.page = None
        self.scraped = None

    async def scrape_available_cams(self):
        """
        Returns scraped available cams.

        :return array of cams dict
        """
        print('\nScrapping cams...\n')

        if (self.browser == None):
            raise Exception('Need to launch browser first')
        
        await self.page.goto('https://globoplay.globo.com/categorias/big-brother-brasil/')

        if (self.scraped == None):
            raise Exception('Unable to perform cam scraping')

        cams = []
        for cam in self.scraped:
            if 'mosaico' not in cam['slug']:
                cams.append(
                    {'name': cam['name'], 'location': cam['media']['headline'], 'snapshot_link': cam['media']['liveThumbnail'], 
                    'slug': cam['slug'], 'media_id': cam['mediaId'],
                    'stream_link': 'https://globoplay.globo.com/'+cam['slug']+'/ao-vivo/'+cam['mediaId']+'/?category=bbb'}
                )
    
        print(str(len(cams)) + ' cams were scraped.\n')

        return cams

    async def response_interceptor(self, response):
        """
        Intercepts response with cams data
        """
        target_url = 'https://cloud-jarvis.globo.com/graphql?operationName=getOfferBroadcastByIdAndAffiliateCode'
        
        try:
            if (response.url.startswith(target_url) and "application/json" in response.headers.get("content-type", "")):
                json_data = await response.json()
                self.scraped = json_data['data']['localizedOffer']['paginatedItems']['resources']
        except json.decoder.JSONDecodeError:
            self.scraped = None

    async def launch_browser(self):
        """
        Launch browser process.
        """
        print('\nLaunching browser...')
        self.browser = await launch()

        # open a new page
        self.page = await self.browser.newPage()

        # disable navigation timeout
        self.page.setDefaultNavigationTimeout(0)
       
        # add response interceptor
        self.page.on('response', 
        lambda response: asyncio.ensure_future(self.response_interceptor(response)))

        print('Browser launched.\n')

    async def close_browser(self):
        """
        Close browser process.
        """
        print('\nClosing browser...')
        await self.browser.close()
        print('Browser closed.\n')

    def scrape_cam_frame(self, snapshot_link):
        """
        Returns frame array according by snapshot link.

        :param snapshot_link: url string of snapshot link

        :return frame array
        """
        resp = requests.get(snapshot_link, stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        return cv2.imdecode(image, cv2.IMREAD_COLOR)