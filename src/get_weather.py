import requests
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# STEP 1: Locate project files
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

air_quality_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "air_quality_clean.csv"
)


# ---------------------------------------------------------
# STEP 2: Read cleaned air-quality data
# ---------------------------------------------------------

air_df = pd.read_csv(
    air_quality_path,
    parse_dates=["time"]
)


# Use the exact same date range as our air-quality data

start_date = air_df["time"].min().date().isoformat()
end_date = air_df["time"].max().date().isoformat()

print("Air-quality period:")
print(start_date, "to", end_date)


# ---------------------------------------------------------
# STEP 3: Request Kathmandu historical weather
# ---------------------------------------------------------

latitude = 27.7172
longitude = 85.3240

weather_url = "https://archive-api.open-meteo.com/v1/archive"

weather_params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m"
    ),
    "timezone": "Asia/Kathmandu"
}


response = requests.get(
    weather_url,
    params=weather_params
)

response.raise_for_status()

weather_json = response.json()


# ---------------------------------------------------------
# STEP 4: Convert JSON to DataFrame
# ---------------------------------------------------------

weather_df = pd.DataFrame(
    weather_json["hourly"]
)

weather_df["time"] = pd.to_datetime(
    weather_df["time"]
)


# ---------------------------------------------------------
# STEP 5: Inspect dataset
# ---------------------------------------------------------

print("\nFirst five weather rows:")
print(weather_df.head())

print("\nWeather dataset shape:")
print(weather_df.shape)

print("\nWeather dataset information:")
weather_df.info()

print("\nMissing values:")
print(weather_df.isnull().sum())


# ---------------------------------------------------------
# STEP 6: Save weather dataset
# ---------------------------------------------------------

weather_output_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "weather_raw.csv"
)

weather_df.to_csv(
    weather_output_path,
    index=False
)

print("\nWeather data saved to:")
print(weather_output_path)