
"""
Example implementation of image model in TensorFlow 
that can be trained and deployed on Cloud ML Engine
"""

import argparse
import json
import os

import trainer.old_model as model
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
        default=0.01,
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
        "--ksize1", help="kernel size of first layer for CNN", type=int, default=5
    )
    parser.add_argument(
        "--ksize2", help="kernel size of second layer for CNN", type=int, default=5
    )
    parser.add_argument(
        "--nfil1", help="number of filters in first layer for CNN", type=int, default=20
    )
    parser.add_argument(
        "--nfil2",
        help="number of filters in second layer for CNN",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dprob", help="dropout probability for CNN", type=float, default=0.25
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

    # Run the training job
    # TODO: replace command line args
    model.train_and_evaluate(output_dir, hparams)
