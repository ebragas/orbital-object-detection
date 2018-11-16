"""
Manages command line arguments for model.py enabling hyperparameter
tuning and other features of Google ML Engine.
"""

import argparse
import json
import os
import trainer.model as model
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--train_batch_size",
        help="Batch size for training steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        help="Initial learning rate for training",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--train_steps",
        help="Steps to run the training job for. A step is one batch-size" "",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )

    # Generate list of model functions to print in help message
    model_names = [
        name.replace("_model", "") for name in dir(model) if name.endswith("_model")
    ]

    parser.add_argument(
        "--model",
        help="Type of model. Supported types are {}".format(model_names),
        default="cnn",
    )
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk",
    )

    # Optional hyperparameters used by cnn
    parser.add_argument(
        "--ksize1", help="kernel size of first layer for CNN", type=int, default=5
    )
    parser.add_argument(
        "--ksize2", help="kernel size of second layer for CNN", type=int, default=5
    )
    parser.add_argument(
        "--nfil1", help="number of filters in first layer for CNN", type=int, default=32
    )
    parser.add_argument(
        "--nfil2",
        help="number of filters in second layer for CNN",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--drop_prob", help="dropout probability for CNN", type=float, default=0.25
    )
    # parser.add_argument(
    #     "--batch_norm",
    #     help="if specified, do batch_norm for CNN",
    #     dest="batch_norm",
    #     action="store_true",
    # )
    # parser.set_defaults(batch_norm=False)

    args = parser.parse_args()
    hparams = args.__dict__

    # Unused args provided by ML Engine
    hparams.pop("job_dir", None)
    hparams.pop("job-dir", None)

    output_dir = hparams.pop("output_dir")

    # Append trial_id to path so hyper-parameter tuning doesn't overwrite each other
    # NOTE: Not 100% clear on how this works yet
    # TODO: Reimplement this later
    output_dir = os.path.join(
        output_dir,
        json.loads(os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", ""),
    )

    # Calculate train_steps if not provided
    if hparams["train_steps"] < 1:
        # 10,000 steps at batch size of 512
        hparams["train_steps"] = (10000 * 512) // hparams["train_batch_size"]
        print("Training for {} steps".format(hparams["train_steps"]))

    # Run the training job
    model.train_and_evaluate(output_dir, hparams)
