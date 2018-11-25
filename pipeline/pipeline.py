#!/usr/bin/env python

from planet_filters import planet_filter
'''
Primary components:
1. Search for additional scenes
2. Write scene metadata to DataStore
3. Download images from metadata; mark entity as downloaded
4. Preprocess and annotate images; store annotated image; mark entity as annotated with image public URL
'''

import os
import json
import requests
import logging
from requests.auth import HTTPBasicAuth
from datetime import datetime
from utils import *
from pprint import pprint

logging.basicConfig(level=logging.DEBUG)


# Planet search filters
logging.info('Loaded planet filters')

if __name__ == "__main__":

    run_start = datetime.now()

    # ----- Planet API ----- #
    filter_name = 'sf_bay'
    item_types = ['PSScene3Band']
    days = 1
    max_cloud_cover = 0.5


    # Get Planet Search endpoint stats
    stats_response = planet_stats_endpoint_request(item_types=item_types,
                                                   filter_name=filter_name,
                                                   days=days,
                                                   max_cloud_cover=max_cloud_cover)

    num_avail_scenes = sum([bucket['count'] for bucket in stats_response['buckets']])

    # Search for scenes
    search_response = planet_search_endpoint_request(item_types=item_types,
                                                     filter_name=filter_name,
                                                     days=days,
                                                     max_cloud_cover=max_cloud_cover)
    
    logging.debug('Planet Search endpoint repsonse {}'.format(search_response))
    
    feature_list = search_response.get('features', [])
    if not feature_list:
        logging.warn('No features were found for the defined search criteria!')
    elif len(feature_list) < num_avail_scenes:
        logging.warn("Additional features are available but were missed because paging isn't implemented yet!")
    
    # Get asset data
    for feature in feature_list:
        feature['assets'] = planet_get_item_assets(item_id=feature['id'], item_type=item_types[0])


    # ----- Loading to DataStore ----- #

    run_end = datetime.now()
    logging.info('\nPipeline completed:\t{}'.format(datetime.now()))
    logging.info('Total runtime:\t{}'.format(run_end - run_start))
