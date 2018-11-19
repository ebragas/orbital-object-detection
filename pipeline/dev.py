#!/usr/bin/env python

""" 
1. Find all the img file names in the dir
2. Parse out the scene_ids
3. Create a dictionary of scenes (key) and labels (w/ lat & lon)
4. Request scene from Planet
5. Write API response and labels to Datastore

# TODO: Cache API responses to file incase of failure during script execution
# TODO: Handle API response errors
# TODO: Check for existing scenes in Datastore before making API requests
"""

from google.cloud import storage
from google.cloud import datastore
from google.cloud.datastore.helpers import GeoPoint
import requests
import os
from time import sleep
from itertools import islice
# import json  # NOTE: dev only

# TODO: Add cmd-line args
PROJECT = "reliable-realm-222318"
DATA_BUCKET = "reliable-realm-222318-vcm"
IMG_DIR = "satellite_imgs/"


def get_blob_names(client, bucket_name, dir_prefix="/"):
    """Get a list of blob names for the given bucket, filter using the prefix.
    Returns list of names as strings

    # TODO: Project name attribute instead of pulling full blob"""

    print('Retrieving blob names... ', end='', flush=True)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blob_names = [
        x.name for x in bucket.list_blobs(prefix=dir_prefix) if x.name.endswith(".png")
    ]

    print('Done.')
    return blob_names


def parse_blob_names(file_names):
    '''Parse scene and label metadata from a list of filenames.

    # TODO: format labels as probas and include class list
    '''
    print('Parsing file names... ', end='', flush=True)
    scenes = {}
    for file_name in file_names:
        direct, name = file_name.rsplit('/', maxsplit=1)  # currently unused
        label, id, coords = name.split('__')
        lon, lat = coords[:-4].split('_')  # slice to ignore file extension

        scenes[id] = scenes.get(id, list())
        scenes[id].append({
            # 'dir': direct,
            # 'file_name': name,
            'label': label,
            'geopoint': GeoPoint(float(lat), float(lon))
        })
    
    print('Done.')
    return scenes


def discard_existing_scenes(scene_dict, client):
    # query for existing keys
    query = client.query(kind='PlanetScenes')
    query.keys_only()
    entities = list(query.fetch())
    entity_keys = [e.key.id_or_name for e in entities]

    new_scene_dict = {k: v for k, v in scene_dict.items() if k not in entity_keys}
    return new_scene_dict


def request_scene_data(session, scenes, item_type):
    '''Request the scene data for scenes dictionary and merge results
    '''
    
    scene_data = []
    total = len(scenes)
    for i, scene_id in enumerate(scenes):
        print('Downloading scene_id: {}; {} of {}... '.format(scene_id, i + 1, total), end='', flush=True)
        url = 'https://api.planet.com/data/v1/item-types/{}/items/{}'.format(item_type, scene_id)
        
        # TODO: implement unsuccessful request handling
        response = session.get(url).json()
        response['labels'] = scenes[scene_id]  # add labels parsed from filename
        # NOTE: assumes we'll only ever get one set of coordinates
        response['geometry']['coordinates'] = [GeoPoint(*coords[::-1]) for coords in response['geometry']['coordinates'][0]]
        scene_data.append(response)
        
        sleep(5)
        print('Done.')
        # break # NOTE: testing only
        
    return scene_data
    

def store_scenes(client, kind, scenes):
    '''TODO: doc'''

    print('Creating {} entities... '.format(len(scenes)), end='', flush=True)
    entity_list = []
    for scene in scenes:
        key = client.key(kind, scene.pop('id'))
        entity = datastore.Entity(key=key)
        entity.update(scene)
        entity_list.append(entity)

    print('Done.')
    print('Writing scenes to Datastore, kind={}'.format(kind), end='', flush=True)
    client.put_multi(entity_list)
    print('Done!')


def batch_dict(d, chunk_size):
    '''Create generator that yields elements of a list in batches of
    size == chunk_size'''

    iterator = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(iterator, chunk_size)}


if __name__ == "__main__":

    # Create Storage client
    storage_client = storage.Client(project=PROJECT)

    # Find all the img file names in the dir
    file_names = get_blob_names(storage_client, DATA_BUCKET, IMG_DIR)

    # Parse data from name
    scenes = parse_blob_names(file_names)

    # Don't try to reload scenes we already have
    datastore_client = datastore.Client(project=PROJECT)
    scenes = discard_existing_scenes(scenes, datastore_client)
    
    # Session and client
    sess = requests.Session()
    sess.auth = (os.environ['PL_API_KEY'], '')

    # Query and store scenes in batches
    for batch in batch_dict(scenes, 10):
        
        # Make API requests
        response_batch = request_scene_data(sess, batch, 'PSScene3Band')

        # Write API response and labels to Datastore
        store_scenes(datastore_client, 'PlanetScenes', response_batch)

    # NOTE: dev only
    # with open('scenes.json', 'w') as out:
    #     json.dump({'data': scenes, 'count': len(scenes)}, out)

    print('Done!')