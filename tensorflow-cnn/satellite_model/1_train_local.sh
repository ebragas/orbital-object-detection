MODEL_DIR=${PWD}/tensorflow-cnn/satellite_model # loc of module and scripts

echo "Deleting old checkpoints..."
rm -r ${MODEL_DIR}/checkpoints

gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${MODEL_DIR}/trainer \
    -- \
    --output_dir=${MODEL_DIR}/checkpoints \
    --model=cnn \
    --train_steps=1 \
    --learning_rate=0.01 \
    --batch_size=1 \
    --augment \
    --train_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/train_data.csv \
    --eval_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/valid_data.csv
