import numpy as np
import tensorflow as tf
import collections
import os

# TODO: Parameterize?
HEIGHT=28
WIDTH=28
CLASSES=tf.constant(['no_ship', 'ship'])

# TODO: I'm not a huge fan of the way I need to create this list right now
#       and parse it into usable lists for training and eval in the get_images
#       function. Would like to spend some time on fixing this
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


def image_parse(features):
    # Pull values from tensor
    filename, label = features['filename'], features['label']
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # Convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, [HEIGHT, WIDTH])

    print(label)
    tf.logging.error('test')
    return image, label

