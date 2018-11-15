"""
Create a CNN using the TensorFlow Estimators API
Created following walkthrough: https://www.tensorflow.org/tutorials/estimators/cnn
"""

# TODO: find out what these are for
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.INFO)

HEIGHT=28
WIDTH=28
NCLASSES=10

# model function
# TODO: Enable hyperparameter tuning; see params dict in https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
def cnn_model_fn(img, mode, params):
    """CNN model function

    Accepts: feature data, labels, and mode specified by tf.estimator.ModeKeys {TRAIN, EVAL, PREDICT}
    """

    # input layer
    input_layer = tf.reshape(img, [-1, HEIGHT, WIDTH, 1])  # expected image size

    # conv. layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv1",
    )

    # pooling layer #1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2, name="pool1"
    )

    # conv. layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv2",
    )

    # pooling layer #2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2, name="pool2"
    )

    # flatten output
    pool2_flat = tf.reshape(
        pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]]) # pool2 height * width * channels

    # dense layer
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense"
    )

    # dropout layer
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )  # TODO: parameterize dropout rate

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=NCLASSES, name="logits")

    return logits, NCLASSES  # NOTE: why do we need to return this here?


def image_classifier(features, labels, mode, params):
    '''Generates estimator spec using the provided hyper-parameters in params.
    '''
    model_fn = cnn_model_fn
    logits, nclasses = model_fn(features['image'], mode, params)

    # Find class prediction and class probabilities
    predictions = {
        # generate predictions for PREDICT and EVAL model
        "classes": tf.argmax(input=logits, axis=1),  # NOTE: do we need tf.cast(..., tf.unit8)?
        # add `softmax_tensor` to graph; used for PREDICT and logging_hook
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # Configure the Training Op for TRAIN mode
    # NOTE: what is an Op?
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # TRAIN
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # NOTE: interesting it takes the logits
        
        # EVAL
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"]
            )
        }

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),  # NOTE: what's a global_step?
            )
        else:
            train_op = None
    
    else:   # NOTE: PREDICT?
        loss = None
        train_op = None
        eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={
            'classes': tf.estimator.export.PredictOutput(predictions)}
    )


def train_and_evaluate(output_dir, hparams):
    """
    Main function for model. Wraps the train_and_evaluate functions and passes through
    hyper-parameters to downstream functions.
    """
    EVAL_INTERVAL = 30  # seconds

    # load training and eval data
    # TODO: Replace with tf.keras.datasets.mnist.load('mnist')
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data = mnist.train.images  # returns np.array
    train_labels = np.asarray(
        mnist.train.labels, dtype=np.int32
    )  # converts array dtype

    eval_data = mnist.test.images  # returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Train Input Function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            "image": train_data
        },  # defined as a dict with key of feature name and value of tensor
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True,
        queue_capacity=5000
    )

    # Eval Input Function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"image": eval_data}, 
        y=eval_labels, 
        batch_size=hparams['train_batch_size'],
        num_epochs=1, 
        shuffle=False,
        queue_capacity=5000
    )

    # create the estimator
    estimator = tf.estimator.Estimator(
        model_fn=image_classifier,
        params=hparams,
        config=tf.estimator.RunConfig(
            save_checkpoints_secs=EVAL_INTERVAL # checkpoint save interval
        ),
        model_dir=output_dir  # location to save checkpoints
    )

    # Train Spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, 
        max_steps=hparams['train_steps']
        )

    # Eval Spec
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=None,
        throttle_secs=EVAL_INTERVAL, name="eval"
    )

    # Train and evaluate loop
    tf.estimator.train_and_evaluate(
        estimator,
        train_spec, 
        eval_spec
        )
