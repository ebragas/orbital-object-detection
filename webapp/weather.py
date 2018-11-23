from datetime import datetime
import os
import pytz
import requests
import math

API_KEY = os.environ['WEATHER_MAP_KEY']
API_URL = 'http://api.openweathermap.org/data/2.5/weather?q={}&mode=json&units=metric&appid={}'

def query_api(city):
    try:
        request_url = API_URL.format(city, API_KEY)
        response = requests.get(request_url).json()

    except Exception as e:
        print(e)
        response = None

    return response

if __name__ == "__main__":
    test_city = 'San Francisco'
    print('Testing API request for {}'.format(test_city))
    print(query_api(test_city))
