
import os
import logging
from google.cloud import datastore
from utils import *

logging.basicConfig(level=logging.DEBUG)

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

    run_start = datetime.now()

    ENTITY_KIND = 'PlanetScenes'
    LIMIT = 20
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
    query.add_filter(f'{ASSET_TYPE}_downloaded', '=', True)
    query.add_filter(f'{ASSET_TYPE}_annotated', '=', False)
    
    result = query.fetch(limit=LIMIT)


    # -------------------- Main Loop ----------------------- #
    
    for entity in result:

        # Download image locally
        entity_id = entity.key.id_or_name
        file_name = f'{ITEM_TYPE}_{entity_id}_{ASSET_TYPE}.tiff' # TODO: use a function for this to enforce consistency
        gs_path = os.path.join(INPUT_DIR, file_name)
        local_path = os.path.join(image_checkpoint_dir, file_name)

        if not os.path.exists(local_path):
            logging.info(f'Downloading image from Cloud Storage location: gs://{BUCKET_NAME}/{gs_path}')
            blob = get_storage_blob(project=PROJECT, bucket_name=BUCKET_NAME, blob_name=gs_path)
            image = download_image_blob(blob)
        else:
            logging.info(f'Reading image from local checkpoint: {local_path}')
            image = Image.open(local_path)

        # Auto-rotate image horizontally
        image = auto_rotate(image)
        
        width, height = image.size
        if height > width:
            image = image.transpose(Image.TRANSPOSE)
        
        # DEV ONLY --> image.save(os.path.join(image_checkpoint_dir, 'sneak_peak.png'), format='PNG')

        # # DEV ONLY -- artificially reduce image size
        # image = image.crop((1000, 1000, 1200, 1200))
        # image.show()

        # Perform object detection
        predictions = maybe_load_from_checkpoint(checkpoint_dir, f'predictions_{entity_id}.json')

        if not predictions:
            bounding_boxes = gen_bounding_box_coords(image, HEIGHT, WIDTH, STEP)

            predictions = perform_object_detection(project_name=PROJECT_NAME,
                                                model_name=MODEL_NAME,
                                                bbox_gen=bounding_boxes,
                                                image=image,
                                                threshold=SAVE_THRESHOLD)
        
            # TODO: cache predictions to tmp at chip level
            write_to_checkpoint(checkpoint_dir, f'predictions_{entity_id}.json', predictions)

        else:
            logging.info('Loading predictions from checkpoint')

        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(image=image,
                                              predictions=predictions,
                                              threshold=DRAW_THRESHOLD)

        today = datetime.today().strftime('%Y%m%d_%H%M%S')
        annotated_name = file_name[:file_name.find('.')] + f'_annotated_{today}.tiff'
        
        # Upload annotated image to Cloud Storage
        upload_image_blob(project=PROJECT_NAME,
                          bucket_name=BUCKET_NAME,
                          dir_prefix=OUTPUT_DIR,
                          blob_name=annotated_name,
                          image=annotated_image,
                          content_type='image/png',
                          format='PNG')

        # Update Entity in DataStore
        entity[f'{ASSET_TYPE}_annotated'] = True
        entity['predictions'] = predictions
        entity['annotated_image_name'] = annotated_name
        datastore_batch_update_entities([entity])

        logging.info('File complete!')


# -------------------------- Wrapping Up ----------------------------- #

    # TODO: Clear tmp dir

    run_end = datetime.now()
    logging.info('')
    logging.info('Pipeline completed:\t{}'.format(datetime.now()))
    logging.info('Total runtime:\t{}'.format(run_end - run_start))
