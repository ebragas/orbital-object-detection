# Purpose
Google Cloud Function that will query Cloud DataStore for Entities which have labels but aren't flagged as having the visual scene downloaded, indicating they were from the original training set. Then downloads these images and updates the corresponding Entity.

# Code
* `main.py` -- contains Cloud Function Python code
* `deploy_function.sh` -- deploys the code to Cloud Function and configuration
* `test_function.sh` -- test cases using curl to make requests to the function HTTP endpoint
