## Set image size the model fraction for the desired model
IMAGE_SIZE=224
# ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"  # mobilenet only
ARCHITECTURE="inception_v3"

## Launch TensorBoard in background
tensorboard --logdir image-retraining/tf_files/training_summaries &

## If TensorBoard is already running, this will fail. Kill it first using:
# pkill -f "tensorboard"

## retrain.py is included in TensorFlow Hub, but not TensorFlow. It was included here for
## simplicity.
## View retrain.py help
python -m image-retraining.scripts.retrain -h

## Retrain model
## NOTE: Also see the image augmentation parameters such as: random_crop, random_scale,
##       random, brightness, etc.
python -m image-retraining.scripts.retrain \
  --bottleneck_dir=image-retraining/tf_files/bottlenecks \
  --how_many_training_steps=15000 \
  --model_dir=image-retraining/tf_files/models/ \
  --summaries_dir=image-retraining/tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=image-retraining/tf_files/retrained_graph.pb \
  --output_labels=image-retraining/tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=image-retraining/tf_files/imgs \
  --learning_rate=0.01 \
  --train_batch_size=25 \
  --validation_batch_size=25 \
  --test_batch_size=25 \
  --architecture=${ARCHITECTURE}
