from ast import literal_eval as make_tuple
from PIL import Image, ImageDraw
from io import BytesIO
import json
import logging
import multiprocessing as mp
import os
import sys
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
from google.cloud import datastore
from google.cloud import storage
from google.cloud.datastore.helpers import GeoPoint
from google.protobuf.json_format import MessageToDict
from itertools import repeat, chain, islice
from time import sleep
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning
# from planet import api # TODO: replace with requests
from oauth2client.client import GoogleCredentials
from io import BytesIO
from googleapiclient import discovery
import base64



logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# TODO: Break these up into subdirectories

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
    # TODO: Update references to `upload_blob_from_filename` and remove
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


def _over_quota():
    logging.error('Stopping script execution as Planet API quota has been reached.')
    sys.exit(0)



### Functions above this line need review and likely to be generalize ###

# ------------------- Image Preprocessing ---------------------- #

def _get_area(img, degree_rotation):
    '''Gets the area of an image that's been rotated the specified number
    of degrees, and extra transparent space cropped
    '''
    img = img.rotate(degree_rotation, expand=True)
    img = img.crop(img.getbbox())
    return np.prod(img.size)


def auto_rotate(image):
    '''Seriously, could I use gradient descent instead?
    '''
    img = image.copy()
    area_arr = np.zeros(45)

    logging.info('Searching for optimal image rotation...')
    for i in range(45):
        area_arr[i] = _get_area(img, i)

    optimal = np.argmin(area_arr)
    logging.info('Optimal rotation degrees: {}'.format(optimal))
    return rotate_crop_image(img, optimal)
        

def parallel_auto_rotate(image, processes=-1):
    '''Iterates over 45 degree range in parallel to find the degree rotation
    of an image that gives the least area
    
    Used to correct rotated images buffered by transparent pixels on sides.
    
    # TODO: make this work or find a better algorithm
    '''
    if processes < 1:
        processes = mp.cpu_count()

    img = image.copy()
    pool = mp.Pool(processes=processes)
    result = pool.starmap(_get_area, zip(repeat(img), range(45)))
    deg = np.argmin(result)

    return rotate_crop_image(img, deg)


def rotate_crop_image(image, degrees):
    '''Returns the rotated image after cropping. Uses expand=True to avoid losing edges
    '''
    im = image.copy()
    im = im.rotate(degrees, expand=True)
    im = im.crop(im.getbbox())
    return im


def gen_bounding_box_coords(image, clip_height, clip_width, step_size):
    '''Returns generator that first yields the total number of bounding boxes given the
    size and step sizes, then returns the bounding box coordinates.
    '''
    coords = []

    # Get original img size
    img_height, img_width = image.size

    num_high = (img_height - (clip_height - step_size)) // step_size
    num_wide = (img_width - (clip_width - step_size)) // step_size

    yield num_high * num_wide

    for i in range(num_high):
        upper = step_size * i
        lower = upper + clip_height

        for j in range(num_wide):
            left = j * step_size
            right = left + clip_width

            yield (left, upper, right, lower)


def draw_bounding_boxes(image, predictions, threshold):
    '''doc'''
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for coord, pred in predictions.items():
        if pred['probabilities'][1] > threshold:
            draw.rectangle(make_tuple(coord), outline='red', width=3)

    return annotated


# ------------------------ ML Engine API ------------------------- #

LOG_INTERVAL = 100  # How many images to classify before logging status
PROJECT_NAME = 'reliable-realm-222318'
MODEL_NAME = 'satellite'

def batch_generator(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator , size - 1))


def classify_image(packet):

    coord = packet['coord']
    body = packet['body']
    
    # Setup clients
    ml = discovery.build('ml', 'v1', cache_discovery=False)
    model_id = 'projects/{}/models/{}'.format(PROJECT_NAME, MODEL_NAME)

    request = ml.projects().predict(name=model_id, body=body)
    response = request.execute()
    
    return coord, response


def exhaust_generator_find_value(generator, find_value):
    # Don't this this will work for my use case
    while True:
        val = next(generator)
        if val == find_value:
            break
    
    return generator


def exhaust_generator_n_times(generator, n_times):
    for i in range(n_times):
        _ = next(generator)
    return generator


def perform_object_detection(project_name, model_name, bbox_gen, image, threshold=0.2):
    '''...
    '''

    # TODO: Work around, cleanup later
    tmp_dir = create_tmp_dir()

    total_bboxes = next(bbox_gen)
    ship_count = 0
    total_count = 0
    cpus = mp.cpu_count() - 1
    logging.info('Multiprocessing using {} CPUs'.format(cpus))
    ckpt_file_name = 'prediction_cache.json'

    # Continue from checkpoint if exists
    predictions = maybe_load_from_checkpoint(tmp_dir, ckpt_file_name)
    
    if predictions:
        logging.info('Continuing prediction from checkpoint')
        total_count = predictions['total_count']
        ship_count = predictions['ship_count']

        bbox_gen = exhaust_generator_n_times(bbox_gen, total_count)

    else:
        predictions = {}

    for coord_batch in batch_generator(bbox_gen, size=cpus):

        # create batch of image chips
        request_queue = []
        for coord in coord_batch:
            clip = image.crop(coord)
            image_bytes = BytesIO()
            clip.save(image_bytes, format='PNG')

            # create batch of requests
            body = {'instances': {'image_bytes': {'b64': base64.b64encode(image_bytes.getvalue()).decode()}}}
            packet ={'coord': coord, 'body': body}
            request_queue.append(packet)

        # multiprocess requests
        logging.debug('Starting multiprocessing... {}'.format(datetime.now()))

        try:
            with mp.get_context("spawn").Pool(initializer=_mute) as pool:
                response_queue = pool.map(classify_image, request_queue)

        except (HttpError, requests.exceptions.SSLError) as e: # Ignore server errors and continue processing
            logging.exception(e)
            logging.warn('Ignoring exception and continueing processing')
            sleep(5)
            continue


        logging.debug('Multiprocessing done. {}'.format(datetime.now()))

        # handle responses
        for coord, response in response_queue:
            for prediction in response['predictions']:
                if prediction['probabilities'][1] > threshold:
                
                    logging.info('Ship detected at {} with {:.2f}% probability'.format(
                        coord, prediction['probabilities'][1]))
                
                    predictions[str(coord)] = prediction
                    ship_count += 1

                total_count += 1

        # Dump current predictions to disk # TODO: figure out how to continue from checkpoints later
        predictions['total_count'] = total_count
        predictions['ship_count'] = ship_count

        write_to_checkpoint(tmp_dir, ckpt_file_name, predictions, update=True)
        
        logging.info('Processed {} images of {}'.format(total_count, total_bboxes))

    logging.info('Total images processed: {}'.format(predictions.pop('total_count')))
    logging.info('Total ships detected: {}'.format(predictions.pop('ship_count')))

    os.rename(src=os.path.join(tmp_dir, ckpt_file_name), dst=os.path.join(tmp_dir, 'prediction_cache_complete{}.json'.format(datetime.now())))

    return predictions


# ------------------------ Planet API ------------------------- #

PL_API_KEY = os.environ['PL_API_KEY']


def planet_build_filter(filter_name='sf_bay', days=1, max_cloud_cover=0.5):
    # TODO: Move these search filters to DataStore
    # TODO: ADD WARNING IF FILTER INCLUDES DATES LESS THAN TWO WEEKS OLD
    #       Planet will give you the feature, but not the assets
    
    # Map parameter to geojson file
    GEOJSON_MAP = {
        'sf_bay': 'geojson/sf_bay.json',
    }

    script_dir = os.path.dirname(os.path.realpath(__file__))
    geojson_path = os.path.join(script_dir, GEOJSON_MAP[filter_name])

    # Load geometry
    with open(geojson_path, 'r') as fp:
        geojson = fp.read()
        geojson = json.loads(geojson)
        geometry = geojson['features'][0]['geometry']

    # Geometry filter
    geometry_filter = {
    'type': 'GeometryFilter',
    'field_name': 'geometry',
    'config': geometry
    }

    # filter for date range
    # FIXME: hardcoded date filters
    date_range_filter = {
        'type': 'DateRangeFilter',
        'field_name': 'acquired',
        'config': {
            # 'gte': '{}'.format((datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")),
            # 'lte': '{}'.format(datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
            'gte': '2018-06-01T11:00:12Z',
            'lte': '2018-06-07T11:00:12Z'
        }
    }

    # cloud cover filter
    cloud_cover_filter = {
        'type': 'RangeFilter',
        'field_name': 'cloud_cover',
        'config': {
            'lte': max_cloud_cover
        }
    }

    # combined filters
    planet_filter = {
        'type': 'AndFilter',
        'config': [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    return planet_filter


def planet_quick_search_endpoint_request(item_types=['PSScene3Band'], filter_name='sf_bay', days=1, max_cloud_cover=0.5):
    
    request_body = {
        'item_types': item_types,
        'filter': planet_build_filter(filter_name, days, max_cloud_cover)
    }
    
    response = requests.post('https://api.planet.com/data/v1/quick-search',
                             auth=HTTPBasicAuth(PL_API_KEY, ''),
                             json=request_body)

    # TODO: implement retry on non 200 response status

    response = response.json()
    logging.debug('Planet Search endpoint repsonse {}'.format(response))

    # Check response quality
    feature_list = response.get('features', [])
    if not feature_list:
        logging.warn('No features were found for the defined search criteria!')
    else:
        logging.info('{} features returned'.format(len(feature_list)))    

    return response


def planet_stats_endpoint_request(item_types=['PSScene3Band'], filter_name='sf_bay', days=1, max_cloud_cover=0.5):
    
    request_body = {
        'interval': 'day',
        'item_types': item_types,
        'filter': planet_build_filter(filter_name, days, max_cloud_cover)
    }

    response = requests.post('https://api.planet.com/data/v1/stats', # TODO: add retry on 500 errors (recursive?)
                             auth=HTTPBasicAuth(PL_API_KEY, ''),
                             json=request_body)

    return response.json()


def planet_get_item_assets(item, item_type):
    '''Get assets using URL specified in item (aka. feature) dictionary
    '''

    assets_url = item.get('_links', {}).get('assets', None)

    # Example: "https://api.planet.com/data/v1/item-types/PSScene3Band/items/20180606_182343_1027/assets/"

    if not assets_url:
        logging.warn('No assets URL specified in item: {}'.format(item['id']))
        return None    

    # Session config. to avoid SSL errors, see note below
    session = requests.Session()
    session.auth = (PL_API_KEY, '')
    session.mount('http', HTTPAdapter(max_retries=3))
    session.mount('https', HTTPAdapter(max_retries=3))

    try:
        response = session.get(assets_url, verify=False).json()
    
    except requests.exceptions.SSLError as exc:
        # FIXME: Don't know why I'm getting these. Tried several options to mitigate, but it's too
        # sporadic to diagnose right now.
        logging.error('SSL error with request: {}'.format(assets_url))
        logging.error('Suppressing error to avoid interrupting processing')
        response = {}

    # FIXME: make this compatible with both JSON and Entities
    # if not response:
    #     logging.warn('No assets returned for item: {}'.format(item['id']))
    # else:
    #     logging.info('{} assets available for item: {}'.format(len(response), item['id']))

    return response


def planet_activate_asset(item, asset_type):
    '''Send activation request to the provided URL'''

    # Check status
    if item['assets'][asset_type]['status'] in ['active', 'activating']:
        logging.info('Asset active and ready to download') # TODO: Add ID somehow
        return
    
    session = requests.Session()
    session.auth = (PL_API_KEY, '')

    # Activate
    activation_url = item['assets'][asset_type]['_links']['activate']
    response = session.post(activation_url)
    
    # Add response to item
    if response.status_code == 202:
        item['assets'][asset_type]['status'] = 'activating'
        logging.info('Activation request successful')

    logging.debug('Activation response code: {}'.format(response.status_code))
    logging.debug('Activation response: {}'.format(response.content))

    return item


def planet_download_asset(item, asset_type, tmp_dir, file_name):
    
    file_path = os.path.join(tmp_dir, file_name)
    session = requests.Session()
    session.auth = (PL_API_KEY, '')

    # Check if active
    if item['assets'][asset_type]['status'] == 'active':
        asset_url = item['assets'][asset_type]['location']
    else:
        logging.warn('Asset not active yet, skipping download')
        return

    # Download
    retry = True
    while retry:
        try:
            response = session.get(asset_url, stream=True, allow_redirects=True)

            # if response.status_code == 400:
            if not response.ok:
                if response.status_code == 400:
                    logging.info('Download token expired. Refreshing and trying again.')
                    item['assets'] = planet_get_item_assets(item, item['properties']['item_type'])
                    asset_url = item['assets'][asset_type]['location']
                else:
                    raise ValueError('Some kinda error, more of a placeholder really')
            else:
                retry = False
        
        except Exception as e:
            # TODO: Handle QuotaExceeded
            raise
    
    if response.status_code == 200:
        logging.info('Request successful. Downloading file...')
        
        with open(file_path, "wb") as fp:
            for chunk in response.iter_content(chunk_size=512):
                if chunk:  # filter out keep-alive new chunks
                    fp.write(chunk)

        logging.info('{} downloaded successfull!'.format(file_name))

        return file_path
        

# --------------------- Google Cloud DateStore ------------------- #

PROJECT = 'reliable-realm-222318'


def datastore_upsert(document, entity_type, entity_id):
    '''Upserts an entity to the specified DataStore collection
    # TODO: wrap in transaction
    '''
    
    client = datastore.Client(project=PROJECT)
    key = client.key(entity_type, entity_id)
    entity = datastore.Entity(key=key)
    
    entity.update(document)
    client.put(entity)


def datastore_batch_upsert_from_json(document_list, entity_type, entity_ids):
    '''Upserts an entity to the specified DataStore collection'''
    
    client = datastore.Client(project=PROJECT)
    
    entity_list = []
    for entity_id, document in zip(entity_ids, document_list):
        key = client.key(entity_type, entity_id)
        entity = datastore.Entity(key=key)
        entity.update(document)
        entity_list.append(entity)
    
    with client.transaction() as xact:
        client.put_multi(entity_list)
        
        # Verify operation
        mutations = xact.mutations
        mutations = [MessageToDict(mut) for mut in mutations]

    # TODO: check all mutations are correct. Is that necessary?
    # logging.debug('Mutations: {}'.format(mutations))
    logging.info('{} mutations completed'.format(len(mutations)))

    return mutations


def datastore_batch_get(entity_kind, entity_names, max_retries=3, wait_secs=30, wait_for_deferred=True):
    '''Kind specifies the collection. Entity names are the values that were used
    to create the unique key
    
    # TODO: add retry count or wait time limit
    '''
    
    client = datastore.Client(project=PROJECT)
    keys = [client.key(kind, name) for kind, name in zip(repeat(entity_kind), entity_names)]
    
    retry = True
    retries = 0
    
    while retry and retries < max_retries:
        deff_entities = []
        entities = client.get_multi(keys, deferred=deff_entities)
        
        if not deff_entities or not wait_for_deferred:
            retry = False
        else:
            sleep(wait_secs)

    return entities


def convert_coord_list_to_geopoints(coordinate_list):
    '''Convert list of coordinate lists like those returned by Planet API for feature geometries
    to GeoPoints objects usable by DataStore
    '''
    geopoints = []
    for lon, lat in coordinate_list:
        geopoints.append(GeoPoint(lat, lon))

    return geopoints


def datastore_batch_update_entities(entities):

    client = datastore.Client(project=PROJECT)

    with client.transaction() as xact:
        client.put_multi(entities)


# ------------------------ Cloud Storage ------------------------- #

def upload_blob_from_filename(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket location
    """
    # NOTE: What happens if this file already exists?

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    logging.info('File {} uploaded to {}.'.format(os.path.basename(source_file_name), destination_blob_name))
    
    return


def get_storage_blobs(project, bucket_name, dir_prefix):
    '''Return list of blob objects from the specified location
    '''
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_prefix))
    return blobs


def get_storage_blob(project, bucket_name, blob_name):
    '''Return a single blob by name
    '''
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blob = bucket.blob(blob_name)
    return blob


def download_image_blob(blob):
    '''Downloads the specified blob and returns as a PIL image object
    '''

    byte_string = blob.download_as_string()
    image_bytes = BytesIO(byte_string)
    image = Image.open(image_bytes)
    return image


def upload_image_blob(project, bucket_name, dir_prefix, blob_name, image, content_type, format='PNG'):
    '''Uploads a blob to the specified location from a file object. This allows uploading of
    BytesIO objects, Images, etc., without the need to first write to disk.

    # TODO: consider handling client in main
    '''
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    
    blob = bucket.blob(blob_name)
    image_bytes = BytesIO()
    image.save(image_bytes, format=format)

    blob.upload_from_string(image_bytes.getvalue(), content_type=content_type)
    logging.info('Uploaded file {} to gs://{}/{}'.format(blob.name, bucket_name, dir_prefix))


# -------------------------- File System --------------------------- #

def create_tmp_dir(directory_name='tmp'):
    '''Create a tmp directory in the current location for caching API repsonses.
    Return the directory path.
    '''
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tmp_dir = os.path.join(script_dir, directory_name)

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    return tmp_dir


def maybe_load_from_checkpoint(tmp_dir, file_name):
    '''Checks if checkpoint file exists and loads if it does. Returns data type based on 
    file_name extension. e.g. *.json returns dict
    '''
    file_path = os.path.join(tmp_dir, file_name)
    checkpoint_exists = os.path.isfile(file_path)
    data = None
    
    if not checkpoint_exists:
        return data

    elif checkpoint_exists and file_name.endswith('.json'):
        with open(file_path) as fp:
            data = json.load(fp)
            logging.info('Loaded {} bytes from checkpoint: {}'.format(os.path.getsize(file_path), file_name))

    elif checkpoint_exists and file_name.endswith('.csv'):
        # FIXME: not used yet; check for headers, index, etc.
        # data = pd.read_csv(file_path)
        raise NotImplementedError

    else:
        raise ValueError("I don't know how to handle this file type yet!: {}".format(file_name.split('.')[-1]))

    return data


def write_to_checkpoint(tmp_dir, file_name, data, update=False):
    '''Write checkpoint file, avoiding overwritting existing files and causing data loss
    '''
    file_path = os.path.join(tmp_dir, file_name)

    # Check if file exists already to prevent overwritting and potential data loss
    if os.path.isfile(file_path) and not update:
        logging.error('File already exists! Raising error to avoid potential data loss!')
        raise ValueError('Checkpoint file already exists: {}'.format(file_path))

    elif file_name.endswith('.json'):
        with open(file_path, 'w') as fp:
            json.dump(data, fp)
            logging.debug('{} bytes written to {}'.format(os.path.getsize(file_path), file_name))

    elif file_name.endswith('.csv'):
        # FIXME: use pandas to write csv with headers but no index
        raise NotImplementedError

    return

def _mute():
    '''Mute logging in child processes'''
    sys.stdout = open(os.devnull, 'w')  