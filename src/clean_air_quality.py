from pathlib import Path
import requests
import pandas as pd


# ---------------------------------------------------------
# STEP 1: Retrieve Kathmandu air-quality data
# ---------------------------------------------------------

latitude = 27.7172
longitude = 85.3240

air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

air_params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": "pm2_5,pm10",
    "timezone": "Asia/Kathmandu",

    # Initial extraction returned 840 rows.
    # 30 historical days should be about 720 hourly rows.
    # The extra rows came from the API's default forecast period.
    # We explicitly remove forecasts because this project
    # is analysing historical observations only.
    "past_days": 30,
    "forecast_days": 0
}


response = requests.get(
    air_quality_url,
    params=air_params
)

response.raise_for_status()

air_json = response.json()

air_df = pd.DataFrame(air_json["hourly"])


# ---------------------------------------------------------
# STEP 2: Fix data types
# ---------------------------------------------------------

# The original extraction stored 'time' as a string.
# Converting it to datetime allows us to analyse
# hours, days, dates, and time-based patterns later.

air_df["time"] = pd.to_datetime(air_df["time"])


# ---------------------------------------------------------
# STEP 3: Validate the dataset
# ---------------------------------------------------------

print("\nFirst five rows:")
print(air_df.head())

print("\nDataset shape:")
print(air_df.shape)

print("\nDataset information:")
air_df.info()

print("\nMissing values:")
print(air_df.isnull().sum())

print("\nDuplicate rows:")
print(air_df.duplicated().sum())

print("\nSummary statistics:")
print(air_df[["pm2_5", "pm10"]].describe())

# ---------------------------------------------------------
# STEP 4: Save cleaned dataset
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

output_path = BASE_DIR / "data" / "processed" / "air_quality_clean.csv"

air_df.to_csv(output_path, index=False)

print(f"\nCleaned dataset saved to:")
print(output_path)  