Google Cloud Functions to perform small singular data ingestion and transformation tasks on a schedule such as ingesting small amount of data at a regular interval from the Planet API.

## Contents
* `main.py` -- contains Cloud Function Python code
* `deploy_function.sh` -- deploys the code to Cloud Function and configuration
* `test_function.sh` -- test cases using curl to make requests to the function HTTP endpoint

## Run
1. Activate Conda environment: `source activate gcp-py37`
2. TODO: include deployment and execution details (bash scripts)

## References
-  [Cloud Functions Runtime](https://cloud.google.com/functions/docs/concepts/python-runtime)