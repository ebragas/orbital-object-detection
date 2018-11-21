MODEL_NAME=satellite
MODEL_VERSION=v01
MODEL_LOCATION=$(gsutil ls gs://reliable-realm-222318-mlengine/satellite_training_181121_133936/output/export/exporter/ | tail -1)
REGION=us-central1
TFVERSION=1.10
PYTHONVERSION=3.5

echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"

# gcloud ml-engine versions delete --quiet ${MODEL_VERSION} --model ${MODEL_NAME}
# gcloud ml-engine models delete ${MODEL_NAME}

# create if doesn't exist
gcloud ml-engine models create ${MODEL_NAME} \
    --regions $REGION \
    --description 'Classify 80x80 images as "ship" or "no ship" from satellite imagery' \
    --enable-logging

# deploy SavedModel as version
gcloud ml-engine versions create ${MODEL_VERSION} \
    --model ${MODEL_NAME} \
    --origin ${MODEL_LOCATION} \
    --runtime-version=$TFVERSION \
    --python-version=$PYTHONVERSION

