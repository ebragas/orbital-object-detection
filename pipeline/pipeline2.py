
import os
import logging
from google.cloud import datastore
from datetime import datetime
from utils import *

# ------------------------------ Setup logging ------------------------------ #
dt = datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

# LOG_FILE = os.path.join(log_dir, 'pipeline2_{}.log'.format(dt))
log_file = os.path.join(log_dir, 'pipeline2.log')

logging.basicConfig(filename=log_file, level=logging.INFO)

# ------------------------------ Settings ------------------------------ #

PROJECT_NAME = 'reliable-realm-222318'
BUCKET_NAME = 'reliable-realm-222318-vcm'
MODEL_NAME = 'satellite'
INPUT_DIR = 'pipeline/full/'
OUTPUT_DIR = os.path.join('pipeline/scenes/annotated/', MODEL_NAME)


# Clip sizes
HEIGHT, WIDTH, STEP = 80, 80, 10

# Prediction thresholds
SAVE_THRESHOLD = 0.5
DRAW_THRESHOLD = 0.9

'''
1.  Query DataStore for Entities that have an 
    associated image file that hasn't already been annotated
2.  Preprocess the image
3.  Perform productions (ideally concurrently), keep only those above theshold
4.  Store annotaed image in GCS
5.  Write predictions back to entity
6.  Write GCS public URL to entity
'''

if __name__ == "__main__":

    try:
        run_start = datetime.now()
        logging.info('Start time: {}'.format(run_start))

        ENTITY_KIND = 'PlanetScenes'
        LIMIT = 10
        ASSET_TYPE = 'visual'
        ITEM_TYPE = 'PSScene3Band'

        # Checkpoint dirs setup
        checkpoint_dir = create_tmp_dir(directory_name='tmp')
        image_checkpoint_dir = create_tmp_dir(os.path.join('tmp/imgs'))
        

        # -------------------- Entity Query -------------------- #

        # Find Entities with images waiting for annotation
        datastore_client = datastore.Client(project=PROJECT_NAME)

        query = datastore_client.query(kind=ENTITY_KIND)
        # query.order = ['-properties.acquired'] # TODO: add when index working
        query.add_filter('{}_downloaded'.format(ASSET_TYPE), '=', True)
        query.add_filter('{}_annotated'.format(ASSET_TYPE), '=', False)
        
        result = query.fetch(limit=LIMIT)


        # -------------------- Main Loop ----------------------- #
        
        for entity in result:

            # Download image locally
            entity_id = entity.key.id_or_name
            file_name = '{}_{}_{}.tiff'.format(ITEM_TYPE, entity_id, ASSET_TYPE) # TODO: use a function for this to enforce consistency
            gs_path = os.path.join(INPUT_DIR, file_name)
            local_path = os.path.join(image_checkpoint_dir, file_name)

            if not os.path.exists(local_path):
                logging.info('Downloading image from Cloud Storage location: gs://{}/{}'.format(BUCKET_NAME, gs_path))
                blob = get_storage_blob(project=PROJECT, bucket_name=BUCKET_NAME, blob_name=gs_path)
                image = download_image_blob(blob)
                image.save(local_path, format='PNG')
            else:
                logging.info('Reading image from local checkpoint: {}'.format(local_path))
                image = Image.open(local_path)

            # Auto-rotate image horizontally
            if not os.path.exists(local_path):
                image = auto_rotate(image)
                
                width, height = image.size
                if height > width:
                    image = image.transpose(Image.TRANSPOSE)

                image.save(local_path, format='PNG')
            
            # NOTE: DEV ONLY --> image.save(os.path.join(image_checkpoint_dir, 'sneak_peak.png'), format='PNG')

            # NOTE: DEV ONLY -- artificially reduce image size
            image = image.crop((1400, 1300, 2400, 2300))
            # image.show()

            bounding_boxes = gen_bounding_box_coords(image, HEIGHT, WIDTH, STEP)

            predictions = perform_object_detection(project_name=PROJECT_NAME,
                                                model_name=MODEL_NAME,
                                                bbox_gen=bounding_boxes,
                                                image=image,
                                                threshold=SAVE_THRESHOLD)
            
            # Draw bounding boxes
            annotated_image = draw_bounding_boxes(image=image,
                                                predictions=predictions,
                                                threshold=DRAW_THRESHOLD)

            today = datetime.today().strftime('%Y%m%d_%H%M%S')
            annotated_name = file_name[:file_name.find('.')] + '_annotated_{}.tiff'.format(today)
            
            # Upload annotated image to Cloud Storage
            upload_image_blob(project=PROJECT_NAME,
                            bucket_name=BUCKET_NAME,
                            dir_prefix=OUTPUT_DIR,
                            blob_name=annotated_name,
                            image=annotated_image,
                            content_type='image/png',
                            format='PNG')

            # Update Entity in DataStore
            entity['{}_annotated'.format(ASSET_TYPE)] = True
            entity['predictions'] = predictions
            entity['annotated_image_name'] = annotated_name
            datastore_batch_update_entities([entity])

            logging.info('File complete!')

    except Exception as e:
        logging.exception(e)
        raise


# -------------------------- Wrapping Up ----------------------------- #

    # TODO: Clear tmp dir

    run_end = datetime.now()
    logging.info('')
    logging.info('Pipeline completed:\t{}'.format(datetime.now()))
    logging.info('Total runtime:\t{}'.format(run_end - run_start))
