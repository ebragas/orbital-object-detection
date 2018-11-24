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

# l = tf.keras.layers  # TODO: merge

tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = "no_ship,ship".split(",") # TODO: replace with: tf.gfile.ListDirectory(os.path.join('/tmp/datasets/dogscats/', 'valid'))
HEIGHT = 224  # TODO: make dynamic when using hub module or basic cnn
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = len(LIST_OF_LABELS) # FIXME: remove; made redundant
EVAL_INTERVAL = 30  # seconds


def tuning_metric(labels, predictions):
    '''Metric used by ML Engine to perform bayesian ____(TODO:?) hyper-parameter tuning.

    Accepts: true labels and predicted labels.
    Returns: a dictionary of key value pairs for the metric name and evaluated metric tensor
        at each row.
    '''
    
    # convert string true label to int
    labels_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(LIST_OF_LABELS)
    )
    labels = labels_table.lookup(labels)
    pred_values = predictions["classid"]

    # TODO: Change to a softmax_cross_entropy_with_logits_v2 loss fn instead
    return {"f1_score": tf.contrib.metrics.f1_score(labels, pred_values)}


def linear_model(img, mode, hparams):
    '''Linear model function; not used'''
    X = tf.reshape(img, [-1, HEIGHT * WIDTH * NUM_CHANNELS])  # flatten
    ylogits = tf.layers.dense(X, NCLASSES, activation=None)
    return ylogits, NCLASSES


def dnn_model(img, mode, hparams):
    '''Deep Neural Network model function; not used'''
    X = tf.reshape(img, [-1, HEIGHT * WIDTH * NUM_CHANNELS])  # flatten
    h1 = tf.layers.dense(X, 300, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
    h3 = tf.layers.dense(h2, 30, activation=tf.nn.relu)
    ylogits = tf.layers.dense(h3, NCLASSES, activation=None)
    return ylogits, NCLASSES


def dnn_dropout_model(img, mode, hparams):
    '''Deep Neural Network with Dropout layers model function; not used'''
    dprob = hparams.get("dprob", 0.1)

    X = tf.reshape(img, [-1, HEIGHT * WIDTH * NUM_CHANNELS])  # flatten
    h1 = tf.layers.dense(X, 300, activation=tf.nn.relu)
    h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
    h3 = tf.layers.dense(h2, 30, activation=tf.nn.relu)
    h3d = tf.layers.dropout(
        h3, rate=dprob, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )  # only dropout when training
    ylogits = tf.layers.dense(h3d, NCLASSES, activation=None)
    return ylogits, NCLASSES


def cnn_model(img, mode, hparams):
    '''A basic Convolutional Neural Network model function using two convolutional
    and pooling layers followed by a dense layer.
    
    Hyper-parameters such as kernel sizes and max pooling filter sizes are configurable
    using hparams argument.

    # TODO (low): Add layer namespaces for grouping in TensorBoard graph
    '''

    # TODO (low): remove; defaults are set in argparse (task.py)
    ksize1 = hparams.get("ksize1", 5)
    ksize2 = hparams.get("ksize2", 5)
    nfil1 = hparams.get("nfil1", 10)
    nfil2 = hparams.get("nfil2", 20)
    dprob = hparams.get("dprob", 0.25)

    # NOTE: CNN doesn't require an input layer because we aren't working with a
    #       feature dictionary as inputs.

    # Convolutional layer 1
    conv1 = tf.layers.conv2d(
        img,
        filters=nfil1,
        kernel_size=ksize1,
        strides=1,
        padding="same",
        activation=tf.nn.relu,
    )
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # Convolutional layer 2
    conv2 = tf.layers.conv2d(
        pool1,
        filters=nfil2,
        kernel_size=ksize2,
        strides=1,
        padding="same",
        activation=tf.nn.relu,
    )
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten layer output
    outlen = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
    pool2flat = tf.reshape(pool2, [-1, outlen])  # flattened

    # Apply batch normalization and dense layer
    if hparams["batch_norm"]:
        h3 = tf.layers.dense(pool2flat, 300, activation=None)
        h3 = tf.layers.batch_normalization(
            h3, training=(mode == tf.estimator.ModeKeys.TRAIN) # only batchnorm when training
        )
        h3 = tf.nn.relu(h3)
    else:
        h3 = tf.layers.dense(pool2flat, 300, activation=tf.nn.relu)

    # Apply dropout
    dropout1 = tf.layers.dropout(
        h3, rate=dprob, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    ylogits = tf.layers.dense(dropout1, NCLASSES, activation=None)

    # Apply batch normalization once more
    if hparams["batch_norm"]:
        ylogits = tf.layers.batch_normalization(
            ylogits, training=(mode == tf.estimator.ModeKeys.TRAIN)
        )

    return ylogits, NCLASSES


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


def serving_input_fn():
    # NOTE: only handles one image at a time; that might be problematic for batch pred.
    feature_placeholders = {'image_bytes': tf.placeholder(tf.string, shape=())}
    image, _ = read_and_preprocess(tf.squeeze(feature_placeholders['image_bytes']))
    image['image'] = tf.expand_dims(image['image'], 0)
    
    return tf.estimator.export.ServingInputReceiver(image, feature_placeholders)


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


def image_classifier(features, labels, mode, params):
    '''Wrapper around the model function.

    Accepts standard arguments per specification for model functions.
        - features: batch of features from input_fn
        - labels: batch of labels from input_fn
        - mode: instance of tf.estimator.ModeKeys
        - params: dict, additional config
    '''

    model_functions = {
        'linear': linear_model,
        'dnn': dnn_model,
        'dnn_dropout': dnn_dropout_model,
        'cnn': cnn_model,
    }

    # Select and run model function
    model_function = model_functions[params["model"]]
    ylogits, nclasses = model_function(features['image'], mode, params)

    # Find predicted values
    probabilities = tf.nn.softmax(ylogits)
    class_int = tf.cast(tf.argmax(probabilities, 1), tf.uint8)
    class_str = tf.gather(LIST_OF_LABELS, tf.cast(class_int, tf.int32))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        # convert string true label to int
        labels_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(LIST_OF_LABELS)
        )
        labels = labels_table.lookup(labels)

        # Compute loss and eval metrics
        # TODO: Research diff. between this and tf.losses.sparse_softmax_cross_entropy
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=ylogits, labels=tf.one_hot(labels, nclasses)
            )
        )

        # Metrics calculated and reported to TensorBoard during evaluation
        eval_metrics = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=class_int, name='acc_op'),
            "recall": tf.metrics.recall(labels=labels, predictions=class_int, name='recall_op'),
            "precision": tf.metrics.precision(labels=labels, predictions=class_int, name='prec_op'),
        }

        # TensorBoard summaries - provides summary metrics to TensorBoard during training
        # TODO (med): check don't need to merge summaries or specify output dir
        for name, metric in eval_metrics.items():
            tf.summary.scalar(name, metric[1])

        # Model update op
        if mode == tf.estimator.ModeKeys.TRAIN:
            # TODO: Try a different optimizer: 
            #       `optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)`
            #       `train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())`
            # NOTE: This is needed for batch normalization, but has no effect otherwise
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contrib.layers.optimize_loss(
                    loss,
                    tf.train.get_global_step(), # Returns total steps taken to optimizer and TensorBoard
                    learning_rate=params["learning_rate"],
                    optimizer="Adam",
                )
        else: # EVAL and PREDICT mode
            train_op = None
    
    else: # PREDICT mode
        loss = None
        train_op = None
        eval_metrics = None

    # Return estimator
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            "probabilities": probabilities,
            "classid": class_int,
            "class": class_str,
        },
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
        export_outputs={
            "classes": tf.estimator.export.PredictOutput(
                {
                    "probabilities": probabilities,
                    "classid": class_int,
                    "class": class_str,
                }
            )
        },
    )


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
    
    NUM_CLASSES = len(params['label_vocab'])
    logit_units = 1 if NUM_CLASSES == 2 else NUM_CLASSES
    logits = tf.layers.Dense(logit_units)(bottleneck_tensor)

    if NUM_CLASSES == 2:
        head = tf.contrib.estimator.binary_classification_head(label_vocabulary=params['label_vocab'])
    else:
        head = tf.contrib.estimator.multi_class_head(n_classes=NUM_CLASSES, label_vocabulary=params['label_vocab'])

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

    # Start up logging
    tf.logging.info('TF Version {}'.format(tf.__version__))
    tf.logging.info('GPU Available {}'.format(tf.test.is_gpu_available()))
    if 'TF_CONFIG' in os.environ:
        tf.logging.info('TF_CONFIG: {}'.format(os.environ["TF_CONFIG"]))

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

    exporter = tf.estimator.LatestExporter("exporter", serving_input_fn)

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
