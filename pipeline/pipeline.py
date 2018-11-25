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
from glob import glob

logging.basicConfig(level=logging.DEBUG)


# Planet search filters
logging.info('Loaded planet filters')

if __name__ == "__main__":

    run_start = datetime.now()

    # ------------------------ Planet API ---------------------------- #

    FILTER_NAME = 'sf_bay'
    ITEM_TYPES = ['PSScene3Band']
    DAYS = 1
    MAX_CLOUD_COVER = 0.5

    # Checkpoint dir
    checkpoint_dir = create_tmp_dir(directory_name='tmp')


    # Get search stats
    stats_response = maybe_load_from_checkpoint(checkpoint_dir, 'stats_response.json')

    if not stats_response: # TODO: improve search by filtering date > newest entity in DataStore
        stats_response = planet_stats_endpoint_request(item_types=ITEM_TYPES,
                                                    filter_name=FILTER_NAME,
                                                    days=DAYS,
                                                    max_cloud_cover=MAX_CLOUD_COVER)

        write_to_checkpoint(checkpoint_dir, 'stats_response.json', stats_response)

    num_avail_scenes = sum([bucket['count'] for bucket in stats_response['buckets']])


    # Get features with search endpoint
    search_response = maybe_load_from_checkpoint(checkpoint_dir, 'search_response.json')

    if not search_response:
        search_response = planet_search_endpoint_request(item_types=ITEM_TYPES,
                                                        filter_name=FILTER_NAME,
                                                        days=DAYS,
                                                        max_cloud_cover=MAX_CLOUD_COVER)
        
        write_to_checkpoint(checkpoint_dir, 'search_response.json', search_response)

    # Check count of returned features
    feature_list = search_response.get('features', [])
    if not feature_list:
        logging.warn('No features were found for the defined search criteria!')
    elif len(feature_list) < num_avail_scenes:
        logging.warn("Additional features are available but were missed because paging isn't implemented yet!")
    

    # Get asset data
    feature_assets = maybe_load_from_checkpoint(checkpoint_dir, 'feature_assets.json')
    
    if not feature_assets:
        for feature in feature_list:
            if not feature.get('assets', {}):
                # TODO: make sure this is working as intended; wasn't returning anything before
                assets = planet_get_item_assets(item_id=feature['id'], item_type=ITEM_TYPES[0])
                feature['assets'] = assets

        # Cache features with assets
        write_to_checkpoint(checkpoint_dir, 'feature_assets.json', feature_list)



    # ------------------------ Load to DataStore ---------------------------- #

    ENT_KIND = 'PlanetScenes'

    # Upsert entity to DataStore
    # TODO: add transactions
    datastore_batch_upsert(feature_list, ENT_KIND, [feature['id'].pop() for feature in feature_list])


    run_end = datetime.now()
    logging.info()
    logging.info('Pipeline completed:\t{}'.format(datetime.now()))
    logging.info('Total runtime:\t{}'.format(run_end - run_start))
