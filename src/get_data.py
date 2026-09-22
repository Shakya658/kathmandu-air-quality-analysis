import requests
import pandas as pd


latitude = 27.7172
longitude = 85.3240


air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "pm2_5,pm10",
    "timezone": "Asia/Kathmandu",
    "past_days": 30
}


response = requests.get(
    air_quality_url,
    params=air_params
)

response.raise_for_status()

air_json = response.json()

air_df = pd.DataFrame(air_json["hourly"])


print(air_df.head())

print("\nDataset shape:")
print(air_df.shape)

print("\nDataset information:")
air_df.info()