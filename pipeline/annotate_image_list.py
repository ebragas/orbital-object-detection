
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

log_file = os.path.join(log_dir, 'pipeline2.log')

logging.basicConfig(filename=log_file, level=logging.INFO)


# ------------------------------ Settings ------------------------------ #

PROJECT_NAME = 'reliable-realm-222318'
BUCKET_NAME = 'reliable-realm-222318-vcm'
MODEL_NAME = 'satellite'
INPUT_DIR = 'pipeline/scenes/raw'
OUTPUT_DIR = 'pipeline/scenes/annotated/test_set'

# Clip sizes
HEIGHT, WIDTH, STEP = 80, 80, 10

# Prediction thresholds
SAVE_THRESHOLD = 0.2
DRAW_THRESHOLD = 0.9

if __name__ == "__main__":

    try:
        run_start = datetime.now()
        logging.info('Start time: {}'.format(run_start))

        # Checkpoint dirs setup
        checkpoint_dir = create_tmp_dir(directory_name='tmp')
        image_checkpoint_dir = create_tmp_dir(os.path.join('tmp/imgs'))        

        # -------------------- Main Loop ----------------------- #
        
        blob_list = get_storage_blobs(PROJECT, BUCKET_NAME, INPUT_DIR)

        for blob in blob_list:
            
            image = download_image_blob(blob)
            
            # # NOTE: DEV ONLY -- artificially reduce image size
            # logging.warn('THIS OPERATION IS INTENDED FOR DEVELOPMENT ONLY!')
            # image = image.crop((0, 1050, 2200, 1350))
            # logging.info('Image size: {}'.format(image.getbbox()))
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

            # # NOTE: DEV ONLY -- artificially reduce image size
            # logging.warn('THIS OPERATION IS INTENDED FOR DEVELOPMENT ONLY!')
            # annotated_image.show()

            today = datetime.today().strftime('%Y%m%d_%H%M%S')
            annotated_name = file_name[:file_name.find('.')] + '_annotated_{}.png'.format(today)
            
            # annotated_image.save(os.path.join(image_checkpoint_dir, annotated_name), format='PNG')
            upload_image_blob(PROJECT_NAME, BUCKET_NAME, OUTPUT_DIR, annotated_name, annotated_image, 'image/png')

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
