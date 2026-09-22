import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# STEP 1: Load analysis-ready dataset
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

data_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "kathmandu_air_weather.csv"
)

df = pd.read_csv(
    data_path,
    parse_dates=["time"]
)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())


# ---------------------------------------------------------
# STEP 2: Create time-based features
# ---------------------------------------------------------

df["date"] = df["time"].dt.date
df["hour"] = df["time"].dt.hour
df["day_name"] = df["time"].dt.day_name()

print("\nDataset with time features:")
print(
    df[
        [
            "time",
            "date",
            "hour",
            "day_name",
            "pm2_5"
        ]
    ].head()
)


# ---------------------------------------------------------
# STEP 3: Average PM2.5 by hour
# ---------------------------------------------------------

hourly_pm25 = (
    df.groupby("hour")["pm2_5"]
    .agg(["mean", "median", "min", "max", "count"])
    .round(2)
)

print("\nPM2.5 statistics by hour:")
print(hourly_pm25)


# ---------------------------------------------------------
# STEP 4: Rank hours by average PM2.5
# ---------------------------------------------------------

highest_pollution_hours = (
    hourly_pm25
    .sort_values(
        "mean",
        ascending=False
    )
)

print("\nHours ranked by average PM2.5:")
print(highest_pollution_hours)


# ---------------------------------------------------------
# STEP 5: Highest individual PM2.5 observations
# ---------------------------------------------------------

highest_pm25_events = (
    df[
        [
            "time",
            "pm2_5",
            "pm10",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m"
        ]
    ]
    .sort_values(
        "pm2_5",
        ascending=False
    )
    .head(10)
)

print("\nTop 10 highest PM2.5 observations:")
print(highest_pm25_events)


# ---------------------------------------------------------
# STEP 6: Daily PM2.5 patterns
# ---------------------------------------------------------

daily_pm25 = (
    df.groupby("date")["pm2_5"]
    .agg(["mean", "min", "max"])
    .round(2)
)

print("\nDaily PM2.5 statistics:")
print(daily_pm25)