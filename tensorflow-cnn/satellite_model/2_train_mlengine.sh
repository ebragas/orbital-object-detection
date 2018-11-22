# Parameters
JOB_NAME=satellite_training_$(date +"%y%m%d_%H%M%S")
# SCALE_TIER=BASIC_GPU  # moved to config.yaml
MODEL_BUCKET=gs://reliable-realm-222318-mlengine
DATA_BUCKET=gs://reliable-realm-222318-vcm
OUTDIR=${MODEL_BUCKET}/${JOB_NAME}              # training output dir
MODEL_DIR=${PWD}/tensorflow-cnn/satellite_model # loc of module and scripts
REGION=us-central1

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --module-name=trainer.task \
    --package-path=${MODEL_DIR}/trainer \
    --job-dir=$OUTDIR/job \
    --config=${MODEL_DIR}/config.yaml \
    -- \
    --output_dir=$OUTDIR/output \
    --train_steps=2000 \
    --learning_rate=0.01 \
    --batch_size=20 \
    --train_data_path=${DATA_BUCKET}/satellite_imgs/csv/train_data.csv \
    --eval_data_path=${DATA_BUCKET}/satellite_imgs/csv/valid_data.csv \
    --augment \
    --batch_norm

# Cancel job
# gcloud ml-engine jobs cancel $JOB_NAME
# gcloud ml-engine jobs list

