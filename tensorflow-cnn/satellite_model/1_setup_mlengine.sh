# Create model
gcloud ml-engine models create satellite \
    --regions=us-central1 \
    --description='Satellite Ship Detector'

# Delete model
# gcloud ml-engine models delete mnist