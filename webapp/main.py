#!/flex-env/bin/env python
from pprint import pprint
import logging
from google.cloud import datastore
from google.cloud import storage

from flask import Flask, flash, redirect, render_template, request, url_for


PROJECT = 'reliable-realm-222318'
BUCKET = 'reliable-realm-222318-vcm'
ENTITY_KIND = 'PlanetScenes'

app = Flask(__name__)

ds_client = datastore.Client(project=PROJECT)
store_client = storage.Client(project=PROJECT)

@app.route('/')
def index():
    """Return home page"""
    return render_template('index.html')

@app.route('/status')
def status_page():
    """Return status page with system stats"""
    
    query = ds_client.query(kind='PlanetScenes')
    query.add_filter('visual_downloaded', '=', True)
    entities = list(query.fetch(limit=None))

    stats = {
        'total_downloaded': len(entities),
        'total_annotated': sum([1 for entity in entities if entity.get('visual_annotated', False) == True])
    }
    return render_template('explorer.html', stats=stats)



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
