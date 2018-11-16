# Parameters
JOB_NAME=satellite_$(date +"%y%m%d_%H%M%S")
SCALE_TIER=BASIC
MODEL_BUCKET=gs://reliable-realm-222318-mlengine
DATA_BUCKET=gs://reliable-realm-222318-vcm
OUTDIR=${MODEL_BUCKET}/${JOB_NAME}
REGION=us-central1

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --region=us-central1 \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/satellite_model/trainer \
    --job-dir=$OUTDIR \
    --scale-tier=$SCALE_TIER \
    --runtime-version=1.8 \
    -- \
    --output_dir=$OUTDIR/output \
    --train_steps=500 \
    --learning_rate=0.01 \
    --batch_size=20 \
    --model=cnn \
    --train_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/train_data.csv \
    --eval_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/valid_data.csv
    --augment \
    --batch_norm

# Cancel job
# gcloud ml-engine jobs cancel $JOB_NAME
# gcloud ml-engine jobs list
