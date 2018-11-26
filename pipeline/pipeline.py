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
from time import sleep
from requests.auth import HTTPBasicAuth
from datetime import datetime
from utils import *
from pprint import pprint
from google.cloud import datastore
from google.cloud.datastore.helpers import GeoPoint

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    run_start = datetime.now()

    # ------------------------ Planet API ---------------------------- #

    FILTER_NAME = 'sf_bay'
    ITEM_TYPES = ['PSScene3Band']
    DAYS = 1
    MAX_CLOUD_COVER = 0.5

    try:

        # Load checkpoints
        checkpoint_dir = create_tmp_dir(directory_name='tmp')
        
        stats_response = maybe_load_from_checkpoint(checkpoint_dir, 'stats_response.json')
        search_response = maybe_load_from_checkpoint(checkpoint_dir, 'search_response.json')
        feature_assets = maybe_load_from_checkpoint(checkpoint_dir, 'feature_assets.json')


        # Get search stats
        if not stats_response: # TODO: improve search by filtering date > newest entity in DataStore
            stats_response = planet_stats_endpoint_request(item_types=ITEM_TYPES,
                                                        filter_name=FILTER_NAME,
                                                        days=DAYS,
                                                        max_cloud_cover=MAX_CLOUD_COVER)

            write_to_checkpoint(checkpoint_dir, 'stats_response.json', stats_response)

        num_avail_scenes = sum([bucket['count'] for bucket in stats_response['buckets']])


        # Get features with search endpoint
        if not search_response:
            search_response = planet_quick_search_endpoint_request(item_types=ITEM_TYPES,
                                                                filter_name=FILTER_NAME,
                                                                days=DAYS,
                                                                max_cloud_cover=MAX_CLOUD_COVER)
            
            write_to_checkpoint(checkpoint_dir, 'search_response.json', search_response)

        # Check response quality
        feature_list = search_response.get('features', [])
        if len(feature_list) < num_avail_scenes:
            logging.warn("Additional features are available but were missed because paging isn't implemented yet!")
        

        # Get item assets
        if not feature_assets:
            feature_assets = feature_list
            
            for feature in feature_assets:
                if not feature.get('assets', {}):
                    # TODO: make sure this is working as intended; wasn't returning anything before
                    assets = planet_get_item_assets(item=feature, item_type=ITEM_TYPES[0])
                    feature['assets'] = assets
                    sleep(3)

            write_to_checkpoint(checkpoint_dir, 'feature_assets.json', feature_list)

    except Exception as e:
        raise



    # ------------------------ Load to DataStore ---------------------------- #

    ENTITY_KIND = 'PlanetScenes'

    try:

        # Convert coordinates to GeoPoints
        for feature in feature_assets:
            feature['geometry']['coordinates'] = convert_coord_list_to_geopoints(
                feature['geometry']['coordinates'][0])

            # NOTE: assumes we only get one geometry, causes intentional data loss otherwise


        # Upsert entities to DataStore
        feature_ids = [feature['id'] for feature in feature_assets]
        datastore_batch_upsert_from_json(feature_assets, ENTITY_KIND, feature_ids)

    except Exception as e:
        raise

    
    # --------------------- Download Available Scenes --------------------- #

    ASSET_TYPE = 'visual'
    BUCKET_NAME = 'reliable-realm-222318-vcm'
    IMAGE_DIR = 'pipeline/full'

    try:

        # Image checkpoint dir
        image_checkpoint_dir = create_tmp_dir(os.path.join('tmp/imgs'))


        # Get previously written entities. Handles voids index update issue.
        entities = datastore_batch_get(ENTITY_KIND, feature_ids)


        # Activate assets, update with new status
        for entity in entities:
            entity = planet_activate_asset(entity, ASSET_TYPE)

        datastore_batch_update_entities(entities)


        # Check asset active and image doesn't already exist
        download_entities = []
        for entity in entities:  # TODO: Mark entities that need to retry download later

            asset_active = entity['assets'][ASSET_TYPE]['status'] == 'active'
            image_exists = entity.get('images', {}).get(ASSET_TYPE, None)
            
            if asset_active and not image_exists:
                download_entities.append(entity)
        

        image_file_paths = []
        for entity in download_entities:
            
            # Check if file checkpointed
            image_file_name = '{}_{}_{}.tiff'.format(ITEM_TYPES[0], entity.key.id_or_name, ASSET_TYPE)
            image_file_path = os.path.join(image_checkpoint_dir, image_file_name)
            
            if not os.path.exists(image_file_path):
                
                # Download image
                image_file_path = planet_download_asset(item=entity,
                                                        asset_type=ASSET_TYPE,
                                                        tmp_dir=image_checkpoint_dir,
                                                        file_name=image_file_name)

            image_file_paths.append(image_file_path)

        
        # Upload image to Cloud Storage and update entity
        for entity, file_path in zip(download_entities, image_file_paths):

            blob_name = os.path.join(IMAGE_DIR, os.path.basename(file_path))

            # Upload to Storage
            upload_blob_from_filename(BUCKET_NAME, file_path, blob_name)

            # Update Entity
            entity['{}_downloaded'.format(ASSET_TYPE)] = True
            entity['{}_annotated'.format(ASSET_TYPE)] = entity.get('{}_annotated'.format(ASSET_TYPE), False)
            datastore_batch_update_entities([entity])

            





    except Exception as e:
        raise


    # -------------------------- Wrapping Up ----------------------------- #

    # TODO: Clear tmp dir

    run_end = datetime.now()
    logging.info('')
    logging.info('Pipeline completed:\t{}'.format(datetime.now()))
    logging.info('Total runtime:\t{}'.format(run_end - run_start))
