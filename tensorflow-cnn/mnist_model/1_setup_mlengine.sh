# Create model
gcloud ml-engine models create mnist \
    --regions=us-central1 \
    --description='MNIST example model'

# Delete model
# gcloud ml-engine models delete mnist