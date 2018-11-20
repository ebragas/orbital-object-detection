import os
import json
import scrapy
from google.cloud import datastore

PROJECT='reliable-realm-222318'

class QuotesSpider(scrapy.Spider):
    name = 'planet'
    http_user = os.environ['PL_API_KEY']
    http_pass = ''

    custom_settings = {
        'IMAGES_EXPIRES': 9999,
        'MEDIA_ALLOW_REDIRECTS': True
    }

    def start_requests(self):
        # query _links for each scene
        client = datastore.Client(project=PROJECT)        
        query = client.query(kind='PlanetScenes')
        query.projection = ['_links.assets']
        results = query.fetch(limit=1)

        # create list of urls for each scene
        asset_urls = [entity['_links.assets'] for entity in results]

        for url in asset_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        # parse json response
        data = json.loads(response.text)

        # check for active visual asset
        if data.get('visual', {}).get('status') == 'active':
            data['image_urls'] = [data['visual']['location']]
        
        else:
            # yield activation
            pass
        
        yield data