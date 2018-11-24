import os
import tensorflow as tf
import tensorflow_hub as hub

l = tf.keras.layers

EVAL_INTERVAL = 30 # seconds

tf.logging.set_verbosity(tf.logging.INFO)



def get_data(local_data_root: str, is_chief: bool=True):
    '''Load sample data locally
    # TODO: remove
    '''
    data_dir = os.path.join(local_data_root, 'datasets/dogscats')
    
    if is_chief:
        if not tf.gfile.IsDirectory(data_dir):
            # Download the data zip to our data directory and extract
            fallback_url = 'http://files.fast.ai/data/dogscats.zip'
            tf.keras.utils.get_file(
                os.path.join(local_data_root, os.path.basename(fallback_url)), 
                fallback_url, 
                cache_dir=local_data_root,
                extract=True)
        
    return data_dir


def _img_string_to_tensor(image_string, image_size=(299, 299)):
    """Decodes jpeg image bytes and resizes into float32 tensor
    
    Args:
      image_string: A Tensor of type string that has the image bytes
    
    Returns:
      float32 tensor of the image
    """
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    image_decoded_as_float = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
    # Resize to expected
    image_resized = tf.image.resize_images(image_decoded_as_float, size=image_size)
    
    return image_resized


def make_dataset(file_pattern, image_size=(299, 299), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096):
    """Makes a dataset reading the input images given the file pattern
    
    Args:
      file_pattern: File pattern to match input files with
      image_size: size to resize images to
      shuffle: whether to shuffle the dataset
      batch_size: the batch size of the dataset
      num_epochs: number of times to repeat iteration of the dataset
      buffer_size: size of buffer for prefetch and shuffle operations
    
    Returns:
      A tf.data.Dataset with dictionary of key to Tensor for features and label Tensor of type string
    """
    
    def _path_to_img(path):
        """From the given path returns a feature dictionary and label pair
        
        Args:
          path: A Tensor of type string of the file path to read from
          
        Returns:
          Tuple of dict and tensor. 
          Dictionary is key to tensor mapping of features
          Label is a Tensor of type string that is the label for these features
        """
        # Get the parent folder of this file to get it's class name
        label = tf.string_split([path], delimiter='/').values[-2]
        
        # Read in the image from disk
        image_string = tf.io.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return { 'image': image_resized }, label
    
    dataset = tf.data.Dataset.list_files(file_pattern)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(_path_to_img)
    dataset = dataset.batch(batch_size).prefetch(buffer_size)

    return dataset


def model_fn(features, labels, mode, params):
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
    logits = l.Dense(logit_units)(bottleneck_tensor)

    if NUM_CLASSES == 2:
        head = tf.contrib.estimator.binary_classification_head(label_vocabulary=params['label_vocab'])
    else:
        head = tf.contrib.estimator.multi_class_head(n_classes=NUM_CLASSES, label_vocabulary=params['label_vocab'])

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get('learning_rate', 1e-3))
    
    return head.create_estimator_spec(
        features, mode, logits, labels, optimizer=optimizer
    )


def train_and_evaluate(output_dir, hparams):

    # Start up logging
    tf.logging.info('TF Version {}'.format(tf.__version__))
    tf.logging.info('GPU Available {}'.format(tf.test.is_gpu_available()))
    if 'TF_CONFIG' in os.environ:
        tf.logging.info('TF_CONFIG: {}'.format(os.environ["TF_CONFIG"]))

    # Load sample data
    # TODO: remove
    data_dir = get_data('/tmp', True)

    # Begin estimator definition, and train/evaluate
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=EVAL_INTERVAL)
    
    data_directory = get_data('/tmp', run_config.is_chief)
    model_directory = '/tmp/dogscats/run2'

    params = {
        'module_spec': 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1',
        'module_name': 'resnet_v2_50',
        'learning_rate': 1e-3,
        'train_module': False,  # Whether we want to finetune the module
        'label_vocab': tf.gfile.ListDirectory(os.path.join(data_directory, 'valid'))
    }

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_directory,
        config=run_config,
        params=params
    )

    input_img_size = hub.get_expected_image_size(hub.Module(params['module_spec']))

    # Train
    train_files = os.path.join(data_directory, 'train', '**/*.jpg')

    def train_input_fn(): return make_dataset(train_files, 
                                              image_size=input_img_size,
                                              batch_size=hparams['batch_size'], 
                                              shuffle=True)
    train_spec = tf.estimator.TrainSpec(
        train_input_fn, 
        max_steps=hparams['train_steps'])

    # Eval
    eval_files = os.path.join(data_directory, 'valid', '**/*.jpg')

    def eval_input_fn(): return make_dataset(eval_files, 
                                             image_size=input_img_size, 
                                             batch_size=hparams['batch_size'])
    
    eval_spec = tf.estimator.EvalSpec(eval_input_fn)

    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
