gcloud ml-engine train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow-cnn/mnist_model/trainer \
    -- \
    --output_dir=output