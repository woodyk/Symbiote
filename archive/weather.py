#!/usr/bin/env python3
#
# weather.py

import requests

def get_weather(location="Dania Beach"):
    try:
        # Format the URL for wttr.in with the specified location
        url = f'https://wttr.in/{location}?format=j1'  # Simple text output with location, weather condition, and temperature
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(response.text)
        else:
            print(f"Failed to get weather data. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
get_weather()

