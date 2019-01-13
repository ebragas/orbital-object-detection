#!/bin/bash

FUNC_NAME="training-ingest"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ENV_VARS="DATA_BUCKET=reliable-realm-222318-vcm,ENTITY_KIND=PlanetScenes,IMAGE_DIR=pipeline/full,PROJECT_NAME=reliable-realm-222318,PL_API_KEY=${PL_API_KEY}"

echo 'Deploying function' ${FUNC_NAME}

gcloud beta functions deploy ${FUNC_NAME} \
    --runtime=python37 \
    --trigger-http \
    --source=${DIR}/${FUNC_NAME} \
    --entry-point=main \
    --timeout=540 \
    --set-env-vars=${ENV_VARS}