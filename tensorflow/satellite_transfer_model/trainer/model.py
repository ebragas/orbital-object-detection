#!/usr/bin/env python

# TODO: Write description
# TODO: Write function doc strings
# TODO: AUPR curve

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_hub as hub


tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = "no_ship,ship".split(",") # TODO: replace with: tf.gfile.ListDirectory(os.path.join('/tmp/datasets/dogscats/', 'valid'))
HEIGHT = 224  # TODO: make dynamic when using hub module or basic cnn
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = len(LIST_OF_LABELS) # FIXME: remove; made redundant
EVAL_INTERVAL = 30  # seconds


def read_and_preprocess_with_augment(image_bytes, label=None):
    return read_and_preprocess(image_bytes, label, augment=True)


def read_and_preprocess(image_bytes, label=None, augment=False):
    # Decode the image
    # End up with pixel values that are in the -1, 1 range
    image = tf.image.decode_jpeg(image_bytes, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 0-1
    image = tf.expand_dims(image, 0)    # resize_bilinear needs batches
                                        # TODO: research resize batches

    # Perform image augmentation
    if augment:
        image = tf.image.resize_bilinear(
            image, [HEIGHT + 10, WIDTH + 10], align_corners=False
        )
        image = tf.squeeze(image)  # remove batch dimension
        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
        
        # NOTE: Commented out because of expectation of consistency in satellite imagery, 
        # could add random rotation instead
        # image = tf.image.random_brightness(image, max_delta=63.0 / 255.0)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        image = tf.image.resize_bilinear(image, [HEIGHT, WIDTH], align_corners=False)
        image = tf.squeeze(image)  # remove batch dimension

    # Standardize pixel values from range (0,1), to (-1,1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    
    return {'image': image}, label


def serving_input_receiver_fn():
    
    feature_spec = {
        'image': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    default_batch_size = 1
    
    serialized_tf_example = tf.placeholder(
        dtype=tf.string, shape=[default_batch_size], 
        name='input_image_tensor')
    
    received_tensors = { 'image': serialized_tf_example }
    
    features = tf.parse_example(serialized_tf_example, feature_spec)
    
    # fn = lambda image: _img_string_to_tensor(image, input_img_size)
    fn = lambda image: _img_string_to_tensor(image, [HEIGHT, WIDTH])
    
    features['image'] = tf.map_fn(fn, features['image'], dtype=tf.float32)
    
    return tf.estimator.export.ServingInputReceiver(features, received_tensors)


def _img_string_to_tensor(image_string, image_size=(299, 299)):
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    
    return image_resized


def make_input_fn(csv_of_filenames, batch_size, mode, augment=False):
    '''
    Function that defines a generic input function whos definition will be configured
    based on the arguments based. The mode argument determines whether the returned
    input function is used for training, eval, or prediction.
    '''
    
    def _input_fn():
        '''Generic input function
        '''

        def decode_csv(csv_row):
            '''
            Reads the CSV row to find the image file path, and reads the image file as bytes.

            Accepts: text line from the CSV file
            Returns: image bytes and label from provided row
            '''
            filename, label = tf.decode_csv(csv_row, record_defaults=[[""], [""]])
            image_bytes = tf.read_file(filename)
            
            return image_bytes, label

        # Create tf.data.dataset from filename
        dataset = tf.data.TextLineDataset(csv_of_filenames).map(decode_csv)

        # Optional image augmentation; use during training only
        if augment:
            dataset = dataset.map(read_and_preprocess_with_augment)
        else:
            dataset = dataset.map(read_and_preprocess)

        # Training dataset settings
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        
        # Eval and predict dataset settings
        else:
            num_epochs = 1  # end-of-input after this

        # Complete dataset and return iterator
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator().get_next()
        
        return iterator

    return _input_fn


def transfer_model_fn(features, labels, mode, params):
    """tf.estimator model function implementation for retraining an image classifier from a 
    tf hub module
    
    Args:
      features: dictionary of key to Tensor
      labels: Tensor of type string
      mode: estimator mode
      params: dictionary of parameters
      
    Returns:
      tf.estimator.EstimatorSpec instance
    """
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    module_trainable = is_training and params.get('train_module', False)

    module = hub.Module(params['module_spec'], trainable=module_trainable, name=params['module_name'])
    bottleneck_tensor = module(features['image'])
    
    num_classes = len(params['label_vocab'])
    logit_units = 1 if num_classes == 2 else num_classes
    logits = tf.layers.Dense(logit_units)(bottleneck_tensor)

    if num_classes == 2:
        head = tf.contrib.estimator.binary_classification_head(label_vocabulary=params['label_vocab'])
    else:
        head = tf.contrib.estimator.multi_class_head(n_classes=num_classes, label_vocabulary=params['label_vocab'])

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 1e-3))
    
    return head.create_estimator_spec(
        features, mode, logits, labels, optimizer=optimizer
    )


def train_and_evaluate(output_dir, hparams):
    '''Main TensorFlow function. Instantiates the model specifications using
    functions defined in this file (model.py), passing the provided hyper-parameters
    where necessary.

    Accepts: output directory for writing checkpoints and summaries, as well as hyper-
        parameter arguments passed at command-line
    
    Instantiates the graph, defining components used for training, evaluation, and prediction,
    then trains the model and produces a model file usable for serving.
    '''

    # Begin estimator definition

    # Hub module params
    # TODO: merge with hparams
    params = {
        'module_spec': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
        'module_name': 'resnet_v2_50',
        'learning_rate': 1e-3,
        'train_module': False,  # Whether we want to finetune the module
        'label_vocab': LIST_OF_LABELS
    }    

    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL)

    # TODO: merge
    estimator = tf.estimator.Estimator(
        model_fn=transfer_model_fn,
        params=params,
        config=run_config,
        model_dir=output_dir,
    )

    # # Hyper-parameter tuning metrics used by ML Engine
    # estimator = tf.contrib.estimator.add_metrics(estimator, tuning_metric)

    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(
            hparams["train_data_path"],
            hparams["batch_size"],
            mode=tf.estimator.ModeKeys.TRAIN,
            # augment=hparams["augment"], # TODO
            augment=False
        ),
        max_steps=hparams["train_steps"],
    )

    exporter = tf.estimator.LatestExporter("exporter", serving_input_receiver_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(
            hparams["eval_data_path"],
            hparams["batch_size"],
            mode=tf.estimator.ModeKeys.EVAL,
            augment=False
        ),
        steps=None,
        exporters=exporter,
        start_delay_secs=EVAL_INTERVAL,
        throttle_secs=EVAL_INTERVAL,
    )

    # Train and evaluate loop
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
