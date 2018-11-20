from google.cloud import datastore
from google.cloud import storage
import logging
import os


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


def get_datastore_ids(project, kind):
    '''Parse scene_ids from entity keys in the specified Datastore entity kind

    # TODO: generalize similarly to `get_storage_ids`
    '''
    logging.info('Querying Datastore PlanetScenes keys...')
    
    client = datastore.Client(project=project)
    query = client.query(kind=kind)
    query.keys_only()
    results = query.fetch()
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


def activate_scene_asset(scene_id):
    pass
