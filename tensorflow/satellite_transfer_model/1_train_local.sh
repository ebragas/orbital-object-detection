MODEL_DIR=${PWD}/tensorflow/satellite_transfer_model # loc of module and scripts

echo "Deleting old checkpoints..."
rm -r ${MODEL_DIR}/checkpoints

gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${MODEL_DIR}/trainer \
    -- \
    --output_dir=${MODEL_DIR}/checkpoints \
    --model=cnn \
    --learning_rate=0.01 \
    --train_steps=50 \
    --batch_size=2 \
    --dense1_nodes=1001 \
    --augment \
    --train_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/train_data.csv \
    --eval_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/valid_data.csv
