from google.cloud import datastore
from google.cloud import storage
from planet import api
import logging
import os
import requests
import sys
import numpy as np
import multiprocessing
from itertools import repeat

def get_blob_names(project, bucket_name, dir_prefix="/"):
    """Get a list of blob names for the given bucket, filter using the prefix.
    Returns list of names as strings

    # TODO: Project name attribute instead of pulling full blob
    """
    logging.info('Querying Cloud Storage blobs...')

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blob_names = [
        x.name for x in bucket.list_blobs(prefix=dir_prefix) if x.name.endswith((".png", "jpg", ".tif"))
    ]

    logging.info('{} blobs returned'.format(len(blob_names)))
    return blob_names


def get_datastore_ids(project, kind, limit=None):
    '''Parse scene_ids from entity keys in the specified Datastore entity kind

    # TODO: generalize similarly to `get_storage_ids`
    '''
    logging.info('Querying Datastore PlanetScenes keys...')
    
    client = datastore.Client(project=project)
    query = client.query(kind=kind)
    query.keys_only()
    results = query.fetch(limit=limit)
    keys = [e.key.id_or_name for e in results]

    logging.info('{} entities returned'.format(len(keys)))
    return keys


def get_storage_ids(project, bucket_name, dir_prefix="/"):
    '''Parse scene_ids from the file names in the specified bucket
    '''
    filenames = get_blob_names(project, bucket_name, dir_prefix)
    
    scene_ids = []
    for path in filenames:
        _, name = path.rsplit('/', maxsplit=1)
        scene_id, _ = name.rsplit('.', maxsplit=1)
        scene_ids.append(scene_id)

    return scene_ids


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    logging.info('File {} uploaded to {}.'.format(os.path.basename(source_file_name), destination_blob_name))


def cleanup(tmp_dir):
    '''Remove files from temp directory
    '''
    for file_name in os.listdir(tmp_dir):
        file_path = os.path.join(tmp_dir, file_name)
        os.remove(file_path)


def get_planet_item(item_id, item_type):
    
    # create client with API Key from PL_API_KEY env. variable
    client = api.ClientV1()
    logging.debug('Planet API Key: {}'.format(client.auth.value))

    try:
        item = client.get_item(item_type, item_id).get()

    except api.exceptions.OverQuota as e:
        _over_quota()
    
    except Exception as e:
        logging.error(e)
        logging.error('Skipping download, item_id {}'.format(item_id))
        item = None

    return item


def maybe_activate_asset(item, asset_type):
    
    # get assets for item
    client = api.ClientV1()
    logging.debug('Planet API Key: {}'.format(client.auth.value))

    # activate asset
    try:
        assets = client.get_assets(item).get()

    except api.exceptions.OverQuota as e:
        _over_quota()
        
    except Exception as e:
        logging.error(e)
        return
    
    if assets.get(asset_type, {}).get('status', '') == 'active':
        # do nothing
        logging.info('{} asset type for item {} already active, ready to download'.format(asset_type.capitalize(), item['id']))
        return


    # NOTE: A response of 202 means that the request has been accepted and the
    # activation will begin shortly. A 204 code indicates that the asset
    # is already active and no further action is needed. A 401 code means
    # the user does not have permissions to download this file.

    # activation request
    try:
        activation = client.activate(assets[asset_type])

    except api.exceptions.OverQuota as e:
        _over_quota()    

    except Exception as e:
        logging.error(e)
        return

    if activation.response.status_code == 202:
        logging.info('Activation request for item_id {} and asset_type {} successful'.format(item['id'], asset_type))
    
    elif activation.response.status_code == 401:
        logging.info('Asset item_id: {} asset_type: {} already activated'.format(item['id'], asset_type))
        logging.debug('If you''re seeing this, thar be bugs!')

    else:
        logging.debug('Unknown activation response status_code: {}'.format(activation.response.status_code))

    return
    

def _over_quota():
    logging.error('Stopping script execution as Planet API quota has been reached.')
    sys.exit(0)


def _get_area(img, degree_rotation):
    '''Gets the area of an image that's been rotated the specified number
    of degrees, and extra transparent space cropped
    '''
    img = img.rotate(degree_rotation, expand=True)
    img = img.crop(img.getbbox())
    return np.prod(img.size)


def parallel_auto_rotate(image, processes=4):
    '''Iterates over 45 degree range in parallel to find the degree rotation
    of an image that gives the least area
    
    Used to correct rotated images buffered by transparent pixels on sides.
    '''
    img = image.copy()
    pool = multiprocessing.Pool(processes=processes)
    result = pool.starmap(_get_area, zip(repeat(img), range(45)))
    return np.argmin(result)