## GCS and AutoML setup
# PROJECT=$(gcloud config get-value project)
#  gcloud projects add-iam-policy-binding $PROJECT \
#     --member="user:ericbragas.4128615@gmail.com" \
#      --role="roles/automl.admin"
#     gcloud projects add-iam-policy-binding $PROJECT \
#        --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
#        --role="roles/ml.admin"
#     gcloud projects add-iam-policy-binding $PROJECT \
#        --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
#        --role="roles/storage.admin"
#     gcloud projects add-iam-policy-binding $PROJECT \
#         --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
#         --role="roles/serviceusage.serviceUsageAdmin"

# Uploads files to GCS recursively
# TODO: upload files to GCS using gsutil
echo "UPLOAD FILES HERE"

# Writes all file contents with label from list of files in GCS
# TODO: make recursive and pull label from dir names\
gsutil ls gs://reliable-realm-222318-sat/automl/ship | awk '{print $1", ship"}' >> ./csv/all_data.csv
gsutil ls gs://reliable-realm-222318-sat/automl/no_ship | awk '{print $1", no_ship"}' >> ./csv/all_data.csv
