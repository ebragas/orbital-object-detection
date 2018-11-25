'''Filter objects used in search of the Planet API
'''

import json
import os
from datetime import datetime

GEOJSON_FILE = 'geojson/sf_bay.json'

# read location geometry
dir_name = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_name, GEOJSON_FILE)) as f:
    location_geojson = f.read()

j = json.loads(location_geojson)
geometry = j['features'][0]['geometry']

# filter for items that overlap with chosen geometry
geometry_filter = {
    'type': 'GeometryFilter',
    'field_name': 'geometry',
    'config': geometry
}

# filter for date range
date_range_filter = {
    'type': 'DateRangeFilter',
    'field_name': 'acquired',
    'config': {
        'gte': '2018-01-01T00:00:00.000Z',
        'lte': '2018-12-31T00:00:00.000Z'
    }
}

# cloud cover filter
cloud_cover_filter = {
    'type': 'RangeFilter',
    'field_name': 'cloud_cover',
    'config': {
        'lte': 0.5
    }
}

# combined filters
planet_filter = {
    'type': 'AndFilter',
    'config': [geometry_filter, date_range_filter, cloud_cover_filter]
}
