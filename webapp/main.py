#!/flex-env/bin/env python
from pprint import pprint
import logging

from flask import Flask, flash, redirect, render_template, request, url_for
from weather import query_api


app = Flask(__name__)


cities = [{'name': 'Toronto'}, {'name': 'Montreal'}, {'name': 'Calgary'},
          {'name': 'Ottawa'}, {'name': 'Edmonton'}, {'name': 'Mississauga'},
          {'name': 'Winnipeg'}, {'name': 'Vancouver'}, {'name': 'Brampton'},
          {'name': 'Quebec'}, {'name': 'San Francisco'}]

@app.route('/')
def index():
    """Return home page"""
    return render_template('weather.html', data=cities)


@app.route('/result', methods=['GET', 'POST'])
def result():
    data = []
    error = None
    select = request.form.get('comp_select') # NOTE: where does this come from?

    response = query_api(select)
    print(response)

    if response:
        data.append(response)
    if len(data) != 2:
        error = 'Bad response from weather API'
    elif not response.get('weather', None):
        error = 'Missing key {}'.format(response.keys())
    return render_template('result.html', data=data, error=error)


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
# [END gae_flex_quickstart]
