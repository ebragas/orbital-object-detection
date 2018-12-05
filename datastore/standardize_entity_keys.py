
# Env: gcp-py37
import os
import logging
from google.cloud import datastore

PROJECT_NAME = 'reliable-realm-222318'
ENTITY_KIND = 'PlanetScenes'

client = datastore.Client(project=PROJECT_NAME)

# Get all entity keys
query = client.query(kind=ENTITY_KIND)
result = list(query.fetch(limit=None))

attribute_names = []

for entity in result:
    for attribute in entity.keys():
        if attribute not in attribute_names:
            attribute_names.append(attribute)

print(attribute_names)

# Set default values

defaults = {'status_msg': None, 
            '_permissions': None, 
            'properties': None, 
            'labels': None,
            '_links': None,
            'geometry': None,
            'type': None,
            'id': None,
            'assets': None,
            'annotated_image_name': None,
            'visual_downloaded': False,
            'visual_annotated': False,
            'predictions': None}

if input('Continue execution and set missing attributes to defaults? [Y/n]: ').lower() == 'y':
    for entity in result:
        for k, v in defaults.items():
            if k not in entity:
                entity[k] = v

    with client.transaction() as xact:
        client.put_multi(result)
        logging.info(f'{len(xact.mutations)} mutations to take place')
        if input('Continue? [Y/n]: ').lower() != 'y':
            xact.rollback()
