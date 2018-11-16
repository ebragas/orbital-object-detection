# Parameters
JOB_NAME=flowers_$(date +"%y%m%d_%H%M%S")
MODEL_BUCKET=gs://reliable-realm-222318-mlengine
DATA_BUCKET=gs://reliable-realm-222318-vcm
OUTDIR=${MODEL_BUCKET}/${JOB_NAME}
REGION=us-central1

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --region=us-central1 \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/flowers_model/trainer \
    --job-dir=$OUTDIR/job \
    --staging-bucket=$MODEL_BUCKET \
    --scale-tier=STANDARD_1 \
    --runtime-version=1.8 \
    -- \
    --output_dir=$OUTDIR/output \
    --train_steps=1000 \
    --learning_rate=0.01 \
    --batch_size=40 \
    --model=cnn \
    --train_data_path=gs://cloud-ml-data/img/flower_photos/train_set.csv \
    --eval_data_path=gs://cloud-ml-data/img/flower_photos/eval_set.csv
    # --augment \
    # --batch_norm \

# Cancel job
# gcloud ml-engine jobs cancel $JOB_NAME
# gcloud ml-engine jobs list