# Submission variables
JOB_NAME=satellite_basic_$(date +"%y%m%d_%H%M%S")
MODEL_BUCKET=gs://reliable-realm-222318-mlengine
DATA_BUCKET=gs://reliable-realm-222318-vcm
OUTDIR=${MODEL_BUCKET}/${JOB_NAME}
REGION=us-central1

echo ${DATA_BUCKET} ${MODEL_BUCKET} ${OUTDIR} ${JOB_NAME}

gsutil -m rm -rf $OUTDIR

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --region=$REGION \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/satellite_model/trainer \
    --job-dir=$OUTDIR/job \
    --staging-bucket=$MODEL_BUCKET \
    --scale-tier=STANDARD_1 \
    --runtime-version=1.10 \
    -- \
    --model=cnn \
    --output_dir=$OUTDIR/output \
    --train_steps=5000 \
    --learning_rate=0.01 \
    --batch_size=40 \
    --nfil1=64 \
    --nfil2=32 \
    --ksize1=5 \
    --ksize2=5 \
    --image_dir=${DATA_BUCKET}/satellite_imgs/csv/all_data.csv

# Cancel job
# gcloud ml-engine jobs cancel $JOB_NAME
# gcloud ml-engine jobs list
