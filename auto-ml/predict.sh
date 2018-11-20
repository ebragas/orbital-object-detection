curl -X POST -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
  https://automl.googleapis.com/v1beta1/projects/reliable-realm-222318/locations/us-central1/models/ICN5377742494555644521:predict -d @request.json