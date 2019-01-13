import logging
import requests
from io import BytesIO
from datetime import datetime
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from google.cloud import datastore
from google.cloud import storage

# TODO: use environment variables (config with gcloud)
PROJECT_NAME = os.environ['PROJECT_NAME']
ENTITY_KIND = os.environ['ENTITY_KIND']
DATA_BUCKET = os.environ['DATA_BUCKET']
IMAGE_DIR = os.environ['IMAGE_DIR']
PL_API_KEY = os.environ['PL_API_KEY']

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def query_entities():
    """
    Queries Cloud DataStore for Entities which have known labels but don't
    have a corresponding image downloaded
    """
    client = datastore.Client(project=PROJECT_NAME)
    query = client.query(kind=ENTITY_KIND)
    query.add_filter("visual_downloaded", "=", False)
    # filter for has labels
    results = query.fetch(limit=None)  # TODO: limit based on quota

    return results


def planet_get_item_assets(item, item_type):
    """Get assets using URL specified in item (aka. feature) dictionary
    """

    assets_url = item.get("_links", {}).get("assets", None)

    # Example: "https://api.planet.com/data/v1/item-types/PSScene3Band/items/20180606_182343_1027/assets/"

    if not assets_url:
        logging.warn("No assets URL specified in item: {}".format(item["id"]))
        return None

    # Session config. to avoid SSL errors, see note below
    session = requests.Session()
    session.auth = (PL_API_KEY, "")
    session.mount("http", HTTPAdapter(max_retries=3))
    session.mount("https", HTTPAdapter(max_retries=3))

    try:
        response = session.get(assets_url, verify=False).json()

    except requests.exceptions.SSLError as exc:
        logging.error("SSL error with request: {}".format(assets_url))
        logging.error("Suppressing error to avoid interrupting processing")
        response = {}

    return response


def main(request=None):
    """Main entry point
    """
    logging.info(f'Starting function execution: {datetime.now()}')

    # Setup Google API clients
    datastore_client = datastore.Client(project=PROJECT_NAME)    
    storage_client = storage.Client(project=PROJECT_NAME)
    bucket = storage_client.get_bucket(bucket_name=DATA_BUCKET)

    # Query DataStore for items requiring download
    entities = query_entities()
    for entity in entities:

        # Get fresh assets
        assets = planet_get_item_assets(entity, "visual")
        
        # Check asset is activated
        if not assets.get("visual", {}).get("status", {}):
            logging.info(
                f"Skipping item {entity.key.id_or_name} as visual asset is not yet active"
            )
            continue

        # Download image
        asset_url = assets["visual"]["location"]
        response = requests.get(asset_url, allow_redirects=True)
        
        if response.status_code == 200:
            data = response.content
            image_bytes = BytesIO(data)

            # Upload to Cloud Storage
            blob_name = "{}_{}_visual.tiff".format(
                entity["properties"]["item_type"], entity.key.id_or_name
            )

            blob = bucket.blob(os.path.join(IMAGE_DIR, blob_name))
            blob.upload_from_string(image_bytes.getvalue(), content_type="image/tiff")

            logging.info("Uploaded file {}.".format(blob.path))

            # Update DataStore Entity image downloaded flag
            entity['visual_downloaded'] = True
            datastore_client.put(entity)
            logging.info(f"Updated entity key({ENTITY_KIND}, '{entity.key.id_or_name}')'")
        
        elif response.status_code == 429:
            logging.warn('Quota Exceeded. Stopping Execution.')
            logging.warn(response.content)
            return
        
        else:
            logging.error(f'Unsucessful request {response}.')


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    
    try:
        main()
    except Exception as e:
        logging.exception(e)
        raise
