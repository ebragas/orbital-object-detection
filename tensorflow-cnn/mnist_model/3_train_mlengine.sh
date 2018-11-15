# Submission variables
JOB_NAME=mnist_$(date +"%y%m%d_%H%M%S")
OUTPUT_PATH=gs://reliable-realm-222318-mlengine/${JOB_NAME}

# Submit job
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir=$OUTPUT_PATH \
    --runtime-version=1.8 \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/mnist_model/trainer \
    --scale-tier=STANDARD_1 \
    -- \
    --output_dir=output \
    --train_steps=20000

echo 'tensorboard --logdir' ${OUTPUT_PATH}


# Cancel job
# gcloud ml-engine jobs cancel $JOB_NAME
# gcloud ml-engine jobs list
