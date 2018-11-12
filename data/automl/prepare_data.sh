## IMPORTANT: Run from the directory ./satellite-object-detection-case-study/data/automl/
## Also make sure to enable billing and the AutoML API as shown here: https://cloud.google.com/vision/automl/docs/quickstart

# Set PROJECT and BUCKET variables
PROJECT=$(gcloud config get-value project) && BUCKET="${PROJECT}-vcm"

# Create bucket if doesn't exist
gsutil mb -p ${PROJECT} -c regional -l us-central1 gs://${BUCKET}

# GCS and AutoML setup
PROJECT=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT \
    --member="user:ericbragas.4128615@gmail.com" \
    --role="roles/automl.admin"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
    --role="roles/ml.admin"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT \
    --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
    --role="roles/serviceusage.serviceUsageAdmin"

# Uploads files to GCS recursively
gsutil -m cp -R ./ gs://${BUCKET}/satellite_imgs

# Writes all file contents with label from list of files in GCS
# TODO: make recursive and pull label from dir names\
mkdir csv
gsutil ls gs://${BUCKET}/satellite_imgs/ship | awk '{print $1", ship"}' >> ./csv/all_data.csv
gsutil ls gs://${BUCKET}/satellite_imgs/no_ship | awk '{print $1", no_ship"}' >> ./csv/all_data.csv

# TODO: filter out .DS_Store files
gsutil cp -R csv/ gs://${BUCKET}/satellite_imgs/csv

echo "CSV file location:"
gsutil ls gs://${BUCKET}/satellite_imgs/csv | head -1