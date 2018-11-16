gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/satellite_model/trainer \
    -- \
    --output_dir=checkpoints \
    --train_steps=500 \
    --learning_rate=0.01 \
    --batch_size=40 \
    --model=cnn \
    --image_dir=gs://reliable-realm-222318-vcm/satellite_imgs/csv/all_data.csv
