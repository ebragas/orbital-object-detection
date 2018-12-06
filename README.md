# Satellite Object Detection Case Study
_Insert helpful info here!_


### Project Structure
* `automl` - bash scripts using `gcloud` for interaction with AutoML service
* `conda-envs` - conda environment exports useful for reproducing the development environment. These were essential to managing all the various dependencies for Google Cloud tools and separate Python runtimes
* `functions` (Python 3.7) - Cloud Functions for performing recurrent data ingest from Planet API. Writes metadata to Cloud DataStore and images to Cloud Storage.
* `data` - ignored directory contains images and other data used for model training, evaluation, and testing
* `dataflow` (Python 2.7) - parallel data pipeline for image preprocessing and prediction using Cloud Dataflow (Apache Beam)
* `datastore` (Python 3.6) - code and bash scripts for defining and working with DataStore service such as defining/managing indexes and data cleanup
* `keras` - Keras models developed for local image classification tasks
* `pipeline` (deprecated) (Python 3.6) - Python scripts for pulling metadata and images on recurring basis from Planet API, performing image preprocessing, making parallel predictions using ML Engine, and pushing results to Cloud DataStore and Cloud Storage.
> This is being replaced in favor of Cloud Functions for recurring data ingest and Cloud DataFlow for scaled up and serverless preprocessing and prediction.
* `scripts` (Python 3.6) - miscellaneous devops scripts
* `tensorflow` (Python 3.6, TensorFlow 1.10) - TensorFlow models for performing image classification. Includes models built and trained from scratch, transfer learning models using TensorFlow Hub (e.g. inceptionV3, VGGNet, ResNetV2, etc.), and TensorFlow `tf.data` pipeline components for fast data ingest.
* `webapp` (Python 3.6) - Flask web application deployed to Cloud App Engine Flexible environment. Enables authenticated users to query database in Cloud DataStore by date, cloud cover, location, etc. for number of cargo ships detected and annotated scene images.


## Conda Environments
Separate Conda environments were used to manage packages and separate Python runtimes. Python 3.5 is used where possible, while Python 2.7 is used primarily for Apache Beam SDK compatiblity.

Environment Name | Runtime | Conda Export File
--- | --- | ---
gcp | Python 3.6 | <env-file.yml>
gcp-dataflow | Python 2.7 | <env-file.yml>
gcp-py37 | Python 3.7 | <env-file.yml>
