## Show script help
python -m image-retraining.scripts.label_image -h

## Test label an image
## ships
# IMG_FILE="image-retraining/tf_files/imgs/ship/1__20160820_233143_0c53__-122.32816455984435_37.73917492083411.jpg"

## not ships
IMG_FILE="image-retraining/tf_files/imgs/no_ship/0__20160622_170157_0c64__-122.34994908648167_37.78274932761714.jpg"

python -m image-retraining.scripts.label_image \
    --graph=image-retraining/tf_files/retrained_graph.pb  \
    --labels=image-retraining/tf_files/retrained_labels.txt \
    --image=${IMG_FILE} \
    --input_height=299 \
    --input_width=299 \
    --input_layer="Mul"

## NOTE: input_height, input_width, and input_layer are set here due to use of inception_v3