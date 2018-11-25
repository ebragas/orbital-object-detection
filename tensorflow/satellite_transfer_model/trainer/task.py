
"""
Example implementation of image model in TensorFlow 
that can be trained and deployed on Cloud ML Engine
"""

import argparse
import json
import os

import trainer.model as model
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Input Arguments
    parser.add_argument(
        "--batch_size", 
        help="Batch size for training steps", 
        type=int, 
        default=100
    )
    parser.add_argument(
        "--learning_rate",
        help="Initial learning rate for training",
        type=float,
        default=0.1e-3,
    )
    parser.add_argument(
        "--train_steps",
        help="Steps to run the training job for. A step is one batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--train_data_path",
        help="location of train file containing eval URLs",
        default="gs://cloud-ml-data/img/flower_photos/train_set.csv",
    )
    parser.add_argument(
        "--eval_data_path",
        help="location of eval file containing img URLs",
        default="gs://cloud-ml-data/img/flower_photos/eval_set.csv",
    )
    # build list of model fn's for help message
    model_names = [
        name.replace("_model", "") for name in dir(model) if name.endswith("_model")
    ]
    parser.add_argument(
        "--model",
        help="Type of model. Supported types are {}".format(model_names),
        default='cnn',
    )
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk",
    )
    parser.add_argument(
        "--augment",
        help="if specified, augment image data",
        dest="augment",
        action="store_true",
    )
    parser.set_defaults(augment=False)

    # optional hyperparameters used by cnn
    parser.add_argument(
        "--dense1_nodes", help="Number of nodes in first dense layer", type=int, default=1001
    )
    parser.add_argument(
        "--dropout_rate", help="dropout probability for CNN", type=float, default=0.25
    )
    parser.add_argument(
        "--batch_norm",
        help="if specified, do batch_norm for CNN",
        dest="batch_norm",
        action="store_true",
    )
    parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    hparams = args.__dict__

    output_dir = hparams.pop("output_dir")
    
    # Append trial_id to path for hptuning
    output_dir = os.path.join(
        output_dir,
        json.loads(os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", ""),
    )

    # Start up logging
    tf.logging.info('TF Version {}'.format(tf.__version__))
    tf.logging.info('GPU Available {}'.format(tf.test.is_gpu_available()))
    if 'TF_CONFIG' in os.environ:
        tf.logging.info('TF_CONFIG: {}'.format(os.environ["TF_CONFIG"]))

    # Run the training job
    model.train_and_evaluate(output_dir, hparams)
