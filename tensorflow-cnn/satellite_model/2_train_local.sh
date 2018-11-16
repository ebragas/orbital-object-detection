gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/satellite_model/trainer \
    -- \
    --output_dir=checkpoints \
    --image_dir=${PWD}/data/imgs \
    --model=cnn