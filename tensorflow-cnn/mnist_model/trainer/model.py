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

# TODO: Replace with command line arg.
cwd = os.getcwd()

# # eager execution
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

# set logging
tf.logging.set_verbosity(tf.logging.INFO)

# model function
# TODO: Enable hyperparameter tuning; see params dict in https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
def cnn_model_fn(features, labels, mode, params):
    """CNN model function

    Accepts: feature data, labels, and mode specified by tf.estimator.ModeKeys {TRAIN, EVAL, PREDICT}
    """

    # input layer
    # NOTE: reshapes image tensors?
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])  # expected image size

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

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # pool2 height * width * channels
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense"
    )
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )  # TODO: parameterize dropout rate

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=10, name="logits")

    # find class prediction and class probabilities
    predictions = {
        # generate predictions for PREDICT and EVAL model
        "classes": tf.argmax(input=logits, axis=1),
        # add `softmax_tensor` to graph; used for PREDICT and logging_hook
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    # calculate loss for TRAIN and EVAL modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # NOTE: interesting it takes the logits

    # configure the Training Op for TRAIN mode
    # NOTE: what is an Op?
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step(),  # NOTE: what's a global_step?
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add eval metrics for EVAL mode
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )


def train_and_evaluate(output_dir, hparams):
    """
    Main function for model. Wraps the train_and_evaluate functions and passes through
    hyper-parameters to downstream functions.
    """
    EVAL_INTERVAL = 30  # seconds

    # load training and eval data
    # TODO: Replace with tf.keras.datasets.mnist.load('mnist')
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    # train_data = mnist.train.images  # returns np.array
    # train_labels = np.asarray(
    #     mnist.train.labels, dtype=np.int32
    # )  # converts array dtype

    # eval_data = mnist.test.images  # returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

    # Train Input Function
    # NOTE: This appears to be a function specifically for creating a training
    #       input function from a Numpy array. Again, we won't have this and
    #       probably wouldn't want to use it anyway since this isn't going to
    #       perform well with large datasets because it reads into memory.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            "x": train_data
        },  # defined as a dict with key of feature name and value of tensor
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True,
    )

    # Eval Input Function
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False
    )

    # create the estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        config=tf.estimator.RunConfig(
            save_checkpoints_secs=EVAL_INTERVAL # checkpoint save interval
        ),
        model_dir=output_dir  # location to save checkpoints
    )

    # Train Spec
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2500)

    # Eval Spec
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=None, throttle_secs=EVAL_INTERVAL, name="eval"
    )

    # Train and evaluate loop
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
