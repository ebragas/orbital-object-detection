import sys
import os
import logging
import base64
from PIL import Image, ImageDraw
from oauth2client.client import GoogleCredentials
from io import BytesIO
from googleapiclient import discovery
from google.cloud import storage
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

PROJECT_NAME = 'reliable-realm-222318'
BUCKET_NAME = 'reliable-realm-222318-vcm'
MODEL_NAME = 'satellite'

INPUT_DIR = 'pipeline/scenes/raw/'
OUTPUT_DIR = 'pipeline/scenes/annotated/'

SAVE_THRESHOLD = 0.2
DRAW_THRESHOLD = 0.9

# Clip sizes
HEIGHT, WIDTH, STEP = 80, 80, 10


def get_storage_blobs(project, bucket_name, dir_prefix):
    '''Return list of blob objects from the specified location
    '''

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_prefix))
    return blobs


def download_image_blob(blob):
    '''Downloads the specified blob and returns as a PIL image object
    '''

    byte_string = blob.download_as_string()
    image_bytes = BytesIO(byte_string)
    image = Image.open(image_bytes)
    return image


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


def perform_object_detection(project_name, model_name, bbox_gen, image, threshold=0.2):
    '''...
    # TODO: implement multi-threading
    '''

    # Setup clients
    ml = discovery.build('ml', 'v1', cache_discovery=False)
    model_id = 'projects/{}/models/{}'.format(project_name, model_name)

    total_bboxes = next(bounding_boxes)
    predictions = {}
    ship_count = 0
    total_count = 0

    for coords in bbox_gen:
        # crop clip
        clip = image.copy()
        clip = clip.crop(coords)

        # save to bytes
        image_bytes = BytesIO()
        clip.save(image_bytes, format='PNG')

        # build and execute request
        # TODO: handle exceptions, retry, etc.
        body = {
            'instances': {
                'image_bytes': {
                    'b64': base64.b64encode(image_bytes.getvalue()).decode()
                }
            }
        }
        request = ml.projects().predict(name=model_id, body=body)
        response = request.execute()

        # handle response
        for prediction in response['predictions']:
            if prediction['probabilities'][1] > threshold:
            
                logging.info('Ship detected at {} with {:.2f}% probability'.format(
                    coords, prediction['probabilities'][1]))
            
                predictions[coords] = prediction
                ship_count += 1

            total_count += 1

        sys.stdout.write('\rProcessed clip: {0} of {1}  '.format(total_count, total_bboxes))

    logging.info('Total images processed: {}'.format(total_count))
    logging.info('Total ships detected: {}'.format(ship_count))

    return predictions


def draw_bounding_boxes(image, predictions, threshold):
    '''doc'''
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for coord, pred in predictions.items():
        if pred['probabilities'][1] > threshold:
            draw.rectangle(coord, outline='red', width=3)

    return annotated


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



if __name__ == '__main__':
    '''Steps:
        1. Access blobs from input dir of GCS bucket
        2. Save blob to image
        3. Break image into clips
        4. Make predictions on clips
        5. Save predictions above a specified threshold
        6. Draw bounding boxes on pos. predictions
        7. Save annotated image to output dir
    '''

    # Pull queue from input dir
    blob_queue = get_storage_blobs(project=PROJECT_NAME,
                                   bucket_name=BUCKET_NAME,
                                   dir_prefix=INPUT_DIR)

    if not blob_queue:
        logging.warn('No files were found at gs://{}/{}'.format(BUCKET_NAME, INPUT_DIR))
    else:
        logging.info('{} files found'.format(len(blob_queue)))

    # Process files individually
    for raw_blob in blob_queue:
        
        logging.info('Downloading {} as image...'.format(raw_blob.name))
        image = download_image_blob(raw_blob)

        # # FIXME: Artificially reduce image size for TESTING ONLY
        # wd, ln = image.size
        # image = image.crop((wd // 2, 0, wd // 1.9, ln // 2))

        bounding_boxes = gen_bounding_box_coords(image, HEIGHT, WIDTH, STEP)

        predictions = perform_object_detection(project_name=PROJECT_NAME,
                                               model_name=MODEL_NAME,
                                               bbox_gen=bounding_boxes,
                                               image=image,
                                               threshold=0.2)
        
        annotated_image = draw_bounding_boxes(image, 
                                              predictions, 
                                              threshold=DRAW_THRESHOLD)

        annotated_blob_name = os.path.join(OUTPUT_DIR, os.path.basename(raw_blob.name).split(
            '.')[0] + '_annotated_' + datetime.now().strftime('%Y%m%d_%H%M%S.png'))

        upload_image_blob(project=PROJECT_NAME,
                          bucket_name=BUCKET_NAME,
                          dir_prefix=OUTPUT_DIR,
                          blob_name=annotated_blob_name,
                          image=annotated_image,
                          content_type=raw_blob.content_type,
                          format='PNG')

        # TODO: Delete completed blobs from input dir, or maybe that's a bad idea... just archive them

        # # FIXME: TESTING ONLY
        # break
    
    print('Dizz-un!')
