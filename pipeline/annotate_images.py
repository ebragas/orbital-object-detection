
import os
import logging
from google.cloud import datastore
from datetime import datetime
from utils import *

# ------------------------------ Setup logging ------------------------------ #
"""
dt = datetime.now().strftime("%Y%m%d_%H%M%S")

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

log_file = os.path.join(log_dir, 'pipeline2.log')

# logging.basicConfig(filename=log_file, level=logging.INFO)
"""
logging.basicConfig(level=logging.INFO)

# ------------------------------ Settings ------------------------------ #

PROJECT_NAME = 'reliable-realm-222318'
BUCKET_NAME = 'reliable-realm-222318-vcm'
MODEL_NAME = 'satellite'
INPUT_DIR = 'pipeline/full/'
OUTPUT_DIR = os.path.join('pipeline/scenes/annotated/', MODEL_NAME)


# Clip sizes
HEIGHT, WIDTH, STEP = 80, 80, 10

# Prediction thresholds
SAVE_THRESHOLD = 0.2
DRAW_THRESHOLD = 0.9

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
        key = datastore_client.key('PlanetScenes', '20180601_182755_0f33')
        result = [datastore_client.get(key)]


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
            logging.warn('THIS OPERATION IS INTENDED FOR DEVELOPMENT ONLY!')
            image = image.crop((4000, 0, 4500, 2620))
            image.show()

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

            # # NOTE: DEV ONLY -- artificially reduce image size
            logging.warn('THIS OPERATION IS INTENDED FOR DEVELOPMENT ONLY!')
            annotated_image.show()

            today = datetime.today().strftime('%Y%m%d_%H%M%S')
            annotated_name = file_name[:file_name.find('.')] + '_annotated_{}.png'.format(today)
            
            annotated_image.save(os.path.join(image_checkpoint_dir, annotated_name), format='PNG')

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
