## Config
PROJECT=reliable-realm-222318   # REPLACE WITH YOUR PROJECT ID
BUCKET=${PROJECT}-mlengine      # REPLACE WITH YOUR BUCKET NAME
REGION=us-central1              # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE=cnn                  # 'linear', 'dnn', 'dnn_dropout', or 'cnn'
TFVERSION=1.12.0

## Delete existing trained model
rm -rf ${PWD}/tensorflow/trained

## Test training locally to ensure ready for ml-engine
gcloud ml-engine local train \
    --module-name=trainer.task \
    --package-path=${PWD}/tensorflow/trainer \
    -- \
    --output_dir=${PWD}/tensorflow/trained \
    --train_steps=100 \
    --learning_rate=0.01 \
    --model=cnn

## Train on ml-engine
OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}
JOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME

gsutil -m rm -rf $OUTDIR

gcloud ml-engine jobs submit training $JOBNAME \
   --region=$REGION \
   --module-name=trainer.task \
   --package-path=${PWD}/tensorflow/trainer \
   --job-dir=$OUTDIR \
   --staging-bucket=gs://$BUCKET \
   --scale-tier=BASIC \
   --runtime-version=$TFVERSION \
   --python-version=3.5 \
   -- \
   --output_dir=$OUTDIR \
   --train_steps=10000 \
   --learning_rate=0.01 \
   --train_batch_size=512 \
   --model=$MODEL_TYPE \
   --batch_norm


## Show TensorBoard
tensorboard --logdir=$OUTDIR
# gs://reliable-realm-222318-mlengine/mnist/trained_cnn