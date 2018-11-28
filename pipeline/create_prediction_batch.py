import os
import shutil
from utils import *
import label_image as lab
import argparse
import tensorflow as tf
import numpy as np

# load the file
FILE_NAME = 'lb_3.png'
OUTPUT_DIR = '/tmp/batch_input/'
PRED_PATH = 'predictions'

"""Sample call
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--input_height=224 --input_width=224 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
"""

model_file = "/tmp/output_graph.pb"
label_file = "/tmp/output_labels.txt"
input_height = 224
input_width = 224
input_mean = 0
input_std = 255
input_layer = "Placeholder"
output_layer = "final_result"

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image to be processed")
parser.add_argument("--graph", help="graph/model to be executed")
parser.add_argument("--labels", help="name of file containing labels")
parser.add_argument("--input_height", type=int, help="input height")
parser.add_argument("--input_width", type=int, help="input width")
parser.add_argument("--input_mean", type=int, help="input mean")
parser.add_argument("--input_std", type=int, help="input std")
parser.add_argument("--input_layer", help="name of input layer")
parser.add_argument("--output_layer", help="name of output layer")
args = parser.parse_args()

if args.graph:
    model_file = args.graph
if args.image:
    file_name = args.image
if args.labels:
    label_file = args.labels
if args.input_height:
    input_height = args.input_height
if args.input_width:
    input_width = args.input_width
if args.input_mean:
    input_mean = args.input_mean
if args.input_std:
    input_std = args.input_std
if args.input_layer:
    input_layer = args.input_layer
if args.output_layer:
    output_layer = args.output_layer

# Create graph
graph = lab.load_graph(model_file)

# clear output dir
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)
tmp_dir = create_tmp_dir()
predictions = {}

image = Image.open('/Users/ericbragas/Downloads/lb_3.png')

# create the bounding boxes
boxes = gen_bounding_box_coords(image, 80, 80, 10)

# crop the image
total_clips = next(boxes)
for batch in batch_generator(boxes, 10):
    clip_paths = []
    for box in batch:
        clip = image.crop(box)
        l, t, r, b = box
        
        # name using the bounding box coords and original image name
        dest_path = os.path.join(OUTPUT_DIR, FILE_NAME[:FILE_NAME.find('.')] + '_{}_{}_{}_{}.jpg'.format(l, t, r, b))
        clip.save(dest_path, format='JPEG')
        clip_paths.append((box, dest_path))
    
    # run the label script on all files
    for coord, file_name in clip_paths:
        t = lab.read_tensor_from_image_file(
            file_name,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name)
        output_operation = graph.get_operation_by_name(output_name)

        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = lab.load_labels(label_file)
        print('{}: {:.4f}, {}'.format(coord, np.max(results), labels[np.argmax(results)]))

        predictions[str(coord)] = {'predictions': results.tolist(), 'class': labels[np.argmax(results)]}
        
        os.remove(file_name)

    write_to_checkpoint(tmp_dir, 'predictions_cache.json', predictions, update=True)

    # read output file and create annotated image

""" 
python label_image.py \
--graph=/tmp/output_graph.pb \
--labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--input_height=224 \
--input_width=224 \
--image=jpeg_imgs/no_ship/0__20161006_002711_0c1b__-122.32556920496363_37.78877724113449.jpg
 """