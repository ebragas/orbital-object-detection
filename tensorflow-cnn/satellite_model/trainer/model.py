from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import os

tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = "no_ship,ship".split(",")
HEIGHT = 80
WIDTH = 80
NUM_CHANNELS = 3
NCLASSES = 2

## Put in utils.py
def create_image_lists(image_dir, test_pct, val_pct):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
      image_dir: String path to a folder containing subfolders of images.
      testing_percentage: Integer percentage of the images to reserve for tests.
      validation_percentage: Integer percentage of images reserved for validation.

    Returns:
      A dictionary containing an entry for each label subfolder, with images split
      into training, testing, and validation sets within each label.
    """
    # Make sure image dir exists
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    
    # Find image subdirectories
    result = collections.OrderedDict()
    sub_dirs = [
        os.path.join(image_dir, item)
        for item in tf.gfile.ListDirectory(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs
                      if tf.gfile.IsDirectory(item))
    
    # Train-test-validation split of images by label
    for sub_dir in sub_dirs:
        extensions = ['png', 'jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)

        # Ignore current dir.
        if dir_name == image_dir:
            continue

        # Find images in dir.
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        tf.logging.info('{} images found.'.format(len(file_list)))
        label_name = dir_name.lower()

        # Split into train, test, validate lists
        test_ind = int(len(file_list) * test_pct)
        vald_ind = test_ind + int(len(file_list) * val_pct)

        testing_images = file_list[:test_ind]
        validation_images = file_list[test_ind:vald_ind]
        training_images = file_list[vald_ind:]

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result

def get_images(img_dict, mode):
    key_map = {tf.estimator.ModeKeys.TRAIN: 'training',
               tf.estimator.ModeKeys.EVAL: 'evaluation'}
    mode_key = key_map[mode]
    filenames = []
    labels = []
    for label in img_dict.keys():
        if mode == tf.estimator.ModeKeys.TRAIN:
            filenames.extend(img_dict[label][mode_key])
            labels.extend([label] * len(img_dict[label][mode_key]))
    return filenames, labels

def decode_images(filename, label):
    # Pull values from tensor
    # filename, label = features['filename'], features['label']
    image_bytes = tf.read_file(filename)
    # image = tf.image.decode_jpeg(image_string, channels=3)
    
    # # Convert to float values in [0, 1]
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize_images(image, [HEIGHT, WIDTH])
    return image_bytes, label

def cnn_model(img, mode, hparams):
    ksize1 = hparams.get("ksize1", 5)
    ksize2 = hparams.get("ksize2", 5)
    nfil1 = hparams.get("nfil1", 64)
    nfil2 = hparams.get("nfil2", 32)
    dprob = hparams.get("dprob", 0.25)

    c1 = tf.layers.conv2d(
        img,
        filters=nfil1,
        kernel_size=ksize1,
        strides=1,
        padding="same",
        activation=tf.nn.relu,
    )
    p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=2)
    c2 = tf.layers.conv2d(
        p1,
        filters=nfil2,
        kernel_size=ksize2,
        strides=1,
        padding="same",
        activation=tf.nn.relu,
    )
    p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=2)

    # p2 HEIGHT * WIDTH * DEPTH
    outlen = p2.shape[1] * p2.shape[2] * p2.shape[3]
    p2flat = tf.reshape(p2, [-1, outlen])  # flattened

    # apply batch normalization
    if hparams["batch_norm"]:
        h3 = tf.layers.dense(p2flat, 300, activation=None)
        h3 = tf.layers.batch_normalization(
            h3, training=(mode == tf.estimator.ModeKeys.TRAIN)
        )  # only batchnorm when training
        h3 = tf.nn.relu(h3)
    else:
        h3 = tf.layers.dense(p2flat, 300, activation=tf.nn.relu)

    # apply dropout
    h3d = tf.layers.dropout(
        h3, rate=dprob, training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    ylogits = tf.layers.dense(h3d, NCLASSES, activation=None)

    # apply batch normalization once more
    if hparams["batch_norm"]:
        ylogits = tf.layers.batch_normalization(
            ylogits, training=(mode == tf.estimator.ModeKeys.TRAIN)
        )

    return ylogits, NCLASSES

def read_and_preprocess_with_augment(image_bytes, label=None):
    return read_and_preprocess(image_bytes, label, augment=True)

def read_and_preprocess(image_bytes, label=None, augment=False):
    # decode the image
    # end up with pixel values that are in the -1, 1 range
    image = tf.image.decode_png(image_bytes, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # 0-1
    image = tf.expand_dims(image, 0)  # resize_bilinear needs batches

    if augment:
        image = tf.image.resize_bilinear(
            image, [HEIGHT + 10, WIDTH + 10], align_corners=False
        )
        image = tf.squeeze(image)  # remove batch dimension
        image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=63.0 / 255.0)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
        image = tf.image.resize_bilinear(image, [HEIGHT, WIDTH], align_corners=False)
        image = tf.squeeze(image)  # remove batch dimension

    # pixel values are in range [0,1], convert to [-1,1]
    # TODO: alternative is to divide by 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return {"image": image}, label

def serving_input_fn():
    # Note: only handles one image at a time
    feature_placeholders = {"image_bytes": tf.placeholder(tf.string, shape=())}
    
    # TODO: research
    image, _ = read_and_preprocess(tf.squeeze(feature_placeholders["image_bytes"]))
    image["image"] = tf.expand_dims(image["image"], 0)
    return tf.estimator.export.ServingInputReceiver(image, feature_placeholders)

def make_input_fn(image_dir, batch_size, mode, augment=False):
    def _input_fn():
        
        # Decode list of images
        def decode_csv(csv_row):
            filename, label = tf.decode_csv(csv_row, record_defaults=[[""], [""]])
            image_bytes = tf.read_file(filename)
            return image_bytes, label

        # Create tf.data.dataset from filename
        dataset = tf.data.TextLineDataset(image_dir).map(decode_csv)

        # Augment
        if augment:
            dataset = dataset.map(read_and_preprocess_with_augment)
        else:
            dataset = dataset.map(read_and_preprocess)

        # Set epochs and shuffling
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            # dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        # Train-eval split
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            valid_dataset = dataset.take(400)
            train_dataset = dataset.skip(400)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return train_dataset.make_one_shot_iterator().get_next()

            elif mode == tf.estimator.ModeKeys.EVAL:
                return valid_dataset.make_one_shot_iterator().get_next()
        
        # Return full dataset if not in train or eval mode
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

def image_classifier(features, labels, mode, params):
    model_functions = {"cnn": cnn_model}

    # Select model
    model_function = model_functions[params["model"]]

    # Forward prop
    ylogits, nclasses = model_function(features["image"], mode, params)

    # Find predicted values
    # TODO: Change to dict
    probabilities = tf.nn.softmax(ylogits)
    class_int = tf.cast(tf.argmax(probabilities, 1), tf.uint8)
    class_str = tf.gather(LIST_OF_LABELS, tf.cast(class_int, tf.int32))


    # Training and Evaluation metrics
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # convert string label to int
        labels_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(LIST_OF_LABELS)
        )
        labels = labels_table.lookup(labels)

        # loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(
        #         logits=ylogits, labels=tf.one_hot(labels, nclasses)
        #     )
        # )

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=ylogits)

        evalmetrics = {"accuracy": tf.metrics.accuracy(class_int, labels)}

        # Backprop operation
        if mode == tf.estimator.ModeKeys.TRAIN:
            # this is needed for batch normalization, but has no effect otherwise
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
            #     train_op = optimizer.minimize(
            #         loss=loss,
            #         global_step=tf.train.get_global_step(),  # NOTE: what's a global_step?
            # )

            optimizer = tf.train.AdamOptimizer(params['learning_rate'])
            train_op = optimizer.minimize(loss)
            
        else:
            train_op = None
    else:
        loss = None
        train_op = None
        evalmetrics = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            "probabilities": probabilities,
            "classid": class_int,
            "class": class_str,
        },
        loss=loss,
        train_op=train_op,
        eval_metric_ops=evalmetrics,
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

def train_and_evaluate(output_dir, hparams):
    EVAL_INTERVAL = 30  # seconds
    
    estimator = tf.estimator.Estimator(
        model_fn=image_classifier,
        params=hparams,
        config=tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL),
        model_dir=output_dir,
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(
            hparams["image_dir"],
            hparams["batch_size"],
            mode=tf.estimator.ModeKeys.TRAIN,
            augment=hparams["augment"],
        ),
        max_steps=hparams["train_steps"],
    )
    
    exporter = tf.estimator.LatestExporter("exporter", serving_input_fn)  # TODO: research
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(
            hparams["image_dir"],
            hparams["batch_size"],
            mode=tf.estimator.ModeKeys.EVAL,
        ),
        steps=None,
        exporters=exporter,
        start_delay_secs=EVAL_INTERVAL,
        throttle_secs=EVAL_INTERVAL,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


