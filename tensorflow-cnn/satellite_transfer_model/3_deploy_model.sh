MODEL_NAME=satellite_transfer
MODEL_VERSION=v02
MODEL_LOCATION=$(gsutil ls gs://reliable-realm-222318-mlengine/satellite_transfer_181117_074615/output/export/exporter | tail -1)
REGION=us-central1
TFVERSION=1.10

echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"

#gcloud ml-engine versions delete --quiet ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}

# create if doesn't exist
# gcloud ml-engine models create ${MODEL_NAME} \
#     --regions $REGION \
#     --description 'Image classification model using transfer learning' \
#     --enable-logging

# deploy SavedModel as version
gcloud ml-engine versions create ${MODEL_VERSION} \
    --model ${MODEL_NAME} \
    --origin ${MODEL_LOCATION} \
    --runtime-version=$TFVERSION \
    --python-version=3.5

