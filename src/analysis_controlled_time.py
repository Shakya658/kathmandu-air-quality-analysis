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

df["hour"] = df["time"].dt.hour

df["rain_status"] = df["precipitation"].apply(
    lambda x: "Rain" if x > 0 else "No rain"
)


# ---------------------------------------------------------
# STEP 2: PM2.5 by hour and rain status
# ---------------------------------------------------------

rain_by_hour = (
    df.groupby(
        ["hour", "rain_status"]
    )["pm2_5"]
    .agg(["mean", "median", "count"])
    .round(2)
    .reset_index()
)

print("\nPM2.5 by hour and rain status:")
print(rain_by_hour)


# ---------------------------------------------------------
# STEP 3: Pivot rain comparison
# ---------------------------------------------------------

rain_pivot = rain_by_hour.pivot(
    index="hour",
    columns="rain_status",
    values="mean"
)

rain_pivot["difference"] = (
    rain_pivot["No rain"]
    - rain_pivot["Rain"]
)

print("\nRain vs no-rain PM2.5 by hour:")
print(
    rain_pivot.round(2)
)


rain_counts = rain_by_hour.pivot(
    index="hour",
    columns="rain_status",
    values="count"
)

print("\nObservation counts by hour:")
print(rain_counts)


# ---------------------------------------------------------
# STEP 4: Summarise rain comparison
# ---------------------------------------------------------

valid_rain_comparisons = (
    rain_pivot
    .dropna(
        subset=["Rain", "No rain"]
    )
)

hours_drier_is_higher = (
    valid_rain_comparisons["difference"] > 0
).sum()

total_comparable_hours = len(
    valid_rain_comparisons
)

median_difference = (
    valid_rain_comparisons["difference"]
    .median()
)

print("\nRain comparison summary:")

print(
    "Hours where dry conditions had higher average PM2.5:",
    hours_drier_is_higher,
    "out of",
    total_comparable_hours
)

print(
    "Median dry-minus-rain PM2.5 difference:",
    round(median_difference, 2)
)


# ---------------------------------------------------------
# STEP 5: Wind relationship within each hour
# ---------------------------------------------------------

wind_results = []

for hour, group in df.groupby("hour"):

    correlation = group[
        "pm2_5"
    ].corr(
        group["wind_speed_10m"]
    )

    wind_results.append(
        {
            "hour": hour,
            "correlation": correlation,
            "count": len(group)
        }
    )


wind_by_hour = pd.DataFrame(
    wind_results
)

print("\nWind-PM2.5 correlation within each hour:")
print(
    wind_by_hour.round(3)
)


negative_wind_hours = (
    wind_by_hour["correlation"] < 0
).sum()

median_wind_correlation = (
    wind_by_hour["correlation"]
    .median()
)

print("\nWind relationship summary:")

print(
    "Hours with negative wind-PM2.5 correlation:",
    negative_wind_hours,
    "out of",
    len(wind_by_hour)
)

print(
    "Median within-hour correlation:",
    round(median_wind_correlation, 3)
)
# ---------------------------------------------------------
# STEP 4B: Keep only reliable rain comparisons
# ---------------------------------------------------------

minimum_observations = 5

reliable_hours = (
    (rain_counts["Rain"] >= minimum_observations)
    &
    (rain_counts["No rain"] >= minimum_observations)
)

reliable_rain_comparisons = (
    rain_pivot[reliable_hours]
)

print(
    "\nReliable rain comparisons "
    f"(at least {minimum_observations} observations in each group):"
)

print(
    reliable_rain_comparisons.round(2)
)

reliable_dry_higher = (
    reliable_rain_comparisons["difference"] > 0
).sum()

reliable_total = len(
    reliable_rain_comparisons
)

reliable_median_difference = (
    reliable_rain_comparisons["difference"]
    .median()
)

print("\nReliable rain comparison summary:")

print(
    "Hours where dry conditions had higher PM2.5:",
    reliable_dry_higher,
    "out of",
    reliable_total
)

print(
    "Median dry-minus-rain difference:",
    round(reliable_median_difference, 2)
)