gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/mnist_model/trainer \
    -- \
    --output_dir=checkpoints