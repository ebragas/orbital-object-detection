# Create model
gcloud ml-engine models create flowers \
    --regions=us-central1 \
    --description='Flowers Multi-class example model'

# Delete model
# gcloud ml-engine models delete flowers