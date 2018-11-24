#!/flex-env/bin/env python
from pprint import pprint
import logging

from flask import Flask, flash, redirect, render_template, request, url_for
from weather import query_api
# from google.cloud import storage


PROJECT = 'reliable-realm-222318'
BUCKET = 'reliable-realm-222318-vcm'

app = Flask(__name__)


def get_storage_blobs(project, bucket_name, dir_prefix):
    '''Return list of blob objects from the specified location
    '''
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name=bucket_name)
    blobs = list(bucket.list_blobs(prefix=dir_prefix))
    return blobs

@app.route('/')
def index():
    """Return home page"""
    return render_template('index.html')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
