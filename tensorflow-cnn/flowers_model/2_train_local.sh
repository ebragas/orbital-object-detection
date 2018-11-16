MODEL=flowers_model

gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/${MODEL}/trainer \
    -- \
    --output_dir=${PWD}/tensorflow-cnn/${MODEL}/checkpoint \
    --train_steps=5 \
    --learning_rate=0.01 \
    --batch_size=2 \
    --model=cnn \
    --augment \
    --train_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/train_data.csv \
    --eval_data_path=gs://reliable-realm-222318-vcm/satellite_imgs/csv/valid_data.csv

    # --train_data_path=gs://cloud-ml-data/img/flower_photos/train_set.csv \
    # --eval_data_path=gs://cloud-ml-data/img/flower_photos/eval_set.csv
