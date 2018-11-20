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

from pipeline.utils import *

PROJECT = 'reliable-realm-222318'
DATA_BUCKET = 'reliable-realm-222318-vcm'
SCENE_DIR = 'pipeline/full/'
TMP_DIR = '/Users/ericbragas/galvanize/satellite-object-detection-case-study/data/tmp'

# API key automatically read from PL_API_KEY env. variable
# client = api.ClientV1()

if __name__ == "__main__":

    # TODO: manage clients in functions
    planet_client = api.ClientV1()
    logging.debug('Planet API Key: {}'.format(planet_client.auth.value))
    downloader = downloader.create(planet_client)
    
    # Get a list of scene_ids from Datastore and GCS
    datastore_scene_ids = get_datastore_ids(project=PROJECT, kind='PlanetScenes')
    storage_scene_ids = get_storage_ids(DATA_BUCKET, SCENE_DIR)
    
    # Download scenes not in that that list of files
    download_queue = []
    for ds_id in datastore_scene_ids:

        # if datastore_scene_id contained in any existing filenames, skip
        if any([x.startswith(ds_id) for x in storage_scene_ids]):
            continue

        # otherwise, queue for download
        download_queue.append(ds_id)
    
    # Request activation of all scene_ids
    for scene_id in download_queue:
        # TODO: activate visual assets
        pass
    
    # Download queued scenes and upload to Cloud Storage
    for scene_id in download_queue:
        
        # Download from Planet
        logging.info('Downloading scene_id: {}'.format(scene_id))
        item = planet_client.get_item('PSScene3Band', scene_id).get()
        
        # Check file not in tmp dir
        if not any([scene_id in file for file in os.listdir(TMP_DIR)]):
            downloader.download(iter([item]), ['visual'], TMP_DIR)
        else:
            logging.info('File found in tmp directory, skipping download')

        # Find downloaded filename
        tmp_file = [x for x in os.listdir(TMP_DIR) if x.startswith(scene_id)][0]

        # Upload to storage
        logging.info('Uploading file to GCS')
        upload_blob(DATA_BUCKET, os.path.join(TMP_DIR, tmp_file), SCENE_DIR + tmp_file)
    
    # logging.info('Cleaning up temp directory')
    # cleanup(TMP_DIR)
    logging.info('All files uploaded! Congrats!')
