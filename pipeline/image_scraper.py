#!/usr/bin/env python

'''Create an image pipeline for Planet API scenes not already in GCS
'''

from google.cloud import datastore
from google.cloud import storage
from planet import api
from planet.api import downloader
from time import sleep
import os, shutil
import logging

logging.basicConfig(level=logging.INFO)

from utils import *

PROJECT = 'reliable-realm-222318'
DATA_BUCKET = 'reliable-realm-222318-vcm'
SCENE_DIR = 'pipeline/full/'
TMP_DIR = './data/tmp'

# API key automatically read from PL_API_KEY env. variable
# client = api.ClientV1()

if __name__ == "__main__":

    # TODO: manage clients in functions
    planet_client = api.ClientV1()
    logging.debug('Planet API Key: {}'.format(planet_client.auth.value))
    downloader = downloader.create(planet_client)
    
    # Get a list of scene_ids from Datastore and GCS
    datastore_scene_ids = get_datastore_ids(project=PROJECT, kind='PlanetScenes', limit=None)
    storage_scene_ids = get_storage_ids(PROJECT, DATA_BUCKET, SCENE_DIR)
    
    # Download scenes not in that that list of files
    # TODO: batch downloading to avoid using up quota on activations of large queue
    download_queue = []
    for ds_id in datastore_scene_ids:

        # if datastore_scene_id contained in any existing filenames, skip
        if any([x.startswith(ds_id) for x in storage_scene_ids]):
            continue

        # otherwise, request items and queue for download
        item = get_planet_item(ds_id, 'PSScene3Band')
        if item:
            download_queue.append(item)
    
    # Request activation of all scene_ids
    for item in download_queue:
        # Activate visual assets
        maybe_activate_asset(item, 'visual')
    
    # Download queued scenes and upload to Cloud Storage
    # TODO: clean-up file exists checks in tmp dir
    for item in download_queue:
        logging.info('Downloading scene_id: {}'.format(item['id']))
        
        # Check file not in tmp dir
        # TODO: move downloader to function and handle exceptions
        if not any([item['id'] in file for file in os.listdir(TMP_DIR)]):
            downloader.download(iter([item]), ['visual'], TMP_DIR)
        else:
            logging.info('File found in tmp directory, skipping download')

        # Find downloaded filename
        # TODO: find a way to get this from downloader
        tmp_file = [x for x in os.listdir(TMP_DIR) if x.startswith(item['id'])][0]

        # Upload to storage
        logging.info('Uploading file to GCS')
        upload_blob(DATA_BUCKET, os.path.join(TMP_DIR, tmp_file), SCENE_DIR + tmp_file)
    
    # logging.info('Cleaning up temp directory')
    # cleanup(TMP_DIR)
    logging.info('All files uploaded! Congrats!')
