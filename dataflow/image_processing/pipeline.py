from __future__ import absolute_import

import sys
import logging
import argparse
from PIL import Image
from io import BytesIO
import base64
# import csv

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from googleapiclient import discovery

HEIGHT = 160
WIDTH = 160
STEP_SIZE = 20



class ReadImageDoFn(beam.DoFn):
    '''Reads image from filename provided by element'''

    def process(self, element):
        filename = element
        image = Image.open(element)

        # NOTE: DEV ONLY
        image = image.crop((0, 1050, 2200, 1350))  # for lb_3.png
        logging.info('Image size: {}'.format(image.size))
        
        return [{'filename': filename, 'image': image}]


class GetChipBoxesDoFn(beam.DoFn):
    '''Returns a list of tuples for bounding box coordinates of an image'''

    def process(self, element):
        boxes = []
        img_height, img_width = element['image'].size

        num_high = (img_height - (HEIGHT - STEP_SIZE)) // STEP_SIZE
        num_wide = (img_width - (WIDTH - STEP_SIZE)) // STEP_SIZE

        logging.info('Total image chips: {}'.format(num_high * num_wide))

        for i in range(num_high):
            upper = STEP_SIZE * i
            lower = upper + HEIGHT

            for j in range(num_wide):
                left = j * STEP_SIZE
                right = left + WIDTH

                yield {'bounding_box': (left, upper, right, lower), 'filename': element['filename'], 'image': element['image']}


class GetPredictionDoFn(beam.DoFn):
    '''Crops an image using the bounding box and requests a prediction'''

    def __init__(self):
        self.PROJECT_NAME = 'reliable-realm-222318'
        self.MODEL_NAME = 'satellite'

    def start_bundle(self, context=None):
        self.ml = discovery.build('ml', 'v1', cache_discovery=False)
        self.model_id = 'projects/{}/models/{}'.format(self.PROJECT_NAME, self.MODEL_NAME)

    def process(self, element):
        image = element['image']
        bounding_box = element['bounding_box']
        filename = element['filename']

        clip = image.crop(bounding_box)
        image_bytes = BytesIO()
        clip.save(image_bytes, format='PNG')
        
        body = {'instances': {'image_bytes': {'b64': base64.b64encode(image_bytes.getvalue()).decode()}}}
        
        request = self.ml.projects().predict(name=self.model_id, body=body)
        response = request.execute()
        
        for pred in response['predictions']:
            if pred['probabilities'][1] > 0.2:
                logging.info('Found ship at {} with {:.2f}% probability'.format(bounding_box, pred['probabilities'][1]))
                yield {'filename': filename, 'bounding_box': bounding_box, 'prediction': pred}
    

def run(argv=None):
    '''Main entry point'''

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input',
                        default='data/csv/scenes.csv',
                        help='Input file containing list of file names')
    
    parser.add_argument('--output',
                        dest='output',
                        default='dataflow/image_processing/formats.csv',
                        help='Output file path')
    
    known_args, pipeline_args = parser.parse_known_args(argv)

    # Pipeline options
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    # Pipeline steps
    images = (p | 'read input file' >> ReadFromText(known_args.input, 
                                                    skip_header_lines=1, 
                                                    strip_trailing_newlines=True)
                | 'read image' >> beam.ParDo(ReadImageDoFn())
                | 'get bounding boxes' >> beam.ParDo(GetChipBoxesDoFn())
                | 'get predictions' >> beam.ParDo(GetPredictionDoFn())
                | 'write to file' >> WriteToText(known_args.output)
                )

    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run(sys.argv[1:])
