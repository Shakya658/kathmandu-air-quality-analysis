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


# ---------------------------------------------------------
# STEP 2: Correlation with PM2.5
# ---------------------------------------------------------

weather_columns = [
    "pm2_5",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m"
]

correlation_matrix = (
    df[weather_columns]
    .corr()
    .round(3)
)

print("\nCorrelation matrix:")
print(correlation_matrix)


pm25_correlations = (
    correlation_matrix["pm2_5"]
    .drop("pm2_5")
    .sort_values()
)

print("\nWeather variables correlated with PM2.5:")
print(pm25_correlations)


# ---------------------------------------------------------
# STEP 3: Compare PM2.5 across wind-speed groups
# ---------------------------------------------------------

df["wind_category"] = pd.cut(
    df["wind_speed_10m"],
    bins=[
        -float("inf"),
        3,
        6,
        10,
        float("inf")
    ],
    labels=[
        "Very low (≤3)",
        "Low (3–6)",
        "Moderate (6–10)",
        "Higher (>10)"
    ]
)

wind_analysis = (
    df.groupby(
        "wind_category",
        observed=True
    )["pm2_5"]
    .agg(["mean", "median", "count"])
    .round(2)
)

print("\nPM2.5 by wind-speed category:")
print(wind_analysis)


# ---------------------------------------------------------
# STEP 4: Compare rainy versus dry hours
# ---------------------------------------------------------

df["rain_status"] = df["precipitation"].apply(
    lambda x: "Rain" if x > 0 else "No rain"
)

rain_analysis = (
    df.groupby("rain_status")["pm2_5"]
    .agg(["mean", "median", "count"])
    .round(2)
)

print("\nPM2.5 during rainy versus dry hours:")
print(rain_analysis)


# ---------------------------------------------------------
# STEP 5: Compare highest 10% pollution observations
# ---------------------------------------------------------

pm25_90th = df["pm2_5"].quantile(0.90)

print("\n90th percentile PM2.5 threshold:")
print(round(pm25_90th, 2))


df["pollution_group"] = (
    df["pm2_5"] >= pm25_90th
)

high_pollution_comparison = (
    df.groupby("pollution_group")[
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m"
        ]
    ]
    .mean()
    .round(2)
)

print("\nAverage weather conditions:")
print("False = lower 90% of PM2.5 observations")
print("True  = highest 10% of PM2.5 observations")
print(high_pollution_comparison)