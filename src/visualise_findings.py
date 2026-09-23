import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# STEP 1: Locate project files
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

data_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "kathmandu_air_weather.csv"
)

figures_dir = (
    BASE_DIR
    / "figures"
)

# Create figures folder if it does not already exist
figures_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# STEP 2: Load analysis-ready dataset
# ---------------------------------------------------------

df = pd.read_csv(
    data_path,
    parse_dates=["time"]
)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# ---------------------------------------------------------
# STEP 3: Create useful time features
# ---------------------------------------------------------

df["hour"] = df["time"].dt.hour
df["date"] = df["time"].dt.date


# =========================================================
# VISUALISATION 1:
# Average PM2.5 by hour of day
# =========================================================

hourly_pm25 = (
    df.groupby("hour")["pm2_5"]
    .mean()
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    hourly_pm25.index,
    hourly_pm25.values,
    marker="o"
)

plt.title(
    "Average PM2.5 by Hour of Day in Kathmandu"
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average PM2.5 (µg/m³)"
)

plt.xticks(
    range(0, 24)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    figures_dir
    / "hourly_pm25_pattern.png",
    dpi=300
)

plt.close()


# =========================================================
# VISUALISATION 2:
# Daily average PM2.5
# =========================================================

daily_pm25 = (
    df.groupby("date")["pm2_5"]
    .mean()
)


plt.figure(
    figsize=(11, 5)
)

plt.plot(
    daily_pm25.index,
    daily_pm25.values,
    marker="o"
)

plt.title(
    "Daily Average PM2.5 in Kathmandu"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Average PM2.5 (µg/m³)"
)

plt.xticks(
    rotation=45
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    figures_dir
    / "daily_pm25_trend.png",
    dpi=300
)

plt.close()


# =========================================================
# VISUALISATION 3:
# Rain versus no rain while controlling for hour
# =========================================================

# Convert precipitation values into two categories:
# Rain and No rain
df["rain_status"] = df["precipitation"].apply(
    lambda x: "Rain"
    if x > 0
    else "No rain"
)


# Calculate average PM2.5 and observation count
# for each hour / rain combination
rain_by_hour = (
    df.groupby(
        [
            "hour",
            "rain_status"
        ]
    )["pm2_5"]
    .agg(
        [
            "mean",
            "count"
        ]
    )
    .reset_index()
)


# ---------------------------------------------------------
# Pivot averages
# ---------------------------------------------------------

# This changes the data from something like:
#
# hour | rain_status | mean
# 0    | Rain        | 29.76
# 0    | No rain     | 37.50
#
# into:
#
# hour | Rain | No rain
# 0    | 29.76| 37.50

rain_mean = rain_by_hour.pivot(
    index="hour",
    columns="rain_status",
    values="mean"
)


# ---------------------------------------------------------
# Pivot observation counts
# ---------------------------------------------------------

rain_count = rain_by_hour.pivot(
    index="hour",
    columns="rain_status",
    values="count"
)


# ---------------------------------------------------------
# Only keep reliable comparisons
# ---------------------------------------------------------

# We require at least 5 observations in BOTH groups.
# This prevents us from comparing averages where,
# for example, one group contains only 1 observation.

minimum_observations = 5

reliable_hours = (
    (rain_count["Rain"] >= minimum_observations)
    &
    (rain_count["No rain"] >= minimum_observations)
)


reliable_rain = (
    rain_mean[
        reliable_hours
    ]
)


# ---------------------------------------------------------
# Reindex to all 24 hours
# ---------------------------------------------------------

# Some hours were removed because they did not have enough
# observations in both groups.
#
# If we plotted reliable_rain directly, Matplotlib would
# connect hour 11 directly to hour 18.
#
# That could visually imply we had observations for
# hours 12-17.
#
# Reindexing creates NaN values for those missing hours.
# Matplotlib breaks the line at NaN values instead.

plot_rain = reliable_rain.reindex(
    range(24)
)


# ---------------------------------------------------------
# Plot rain vs no-rain comparison
# ---------------------------------------------------------

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    plot_rain.index,
    plot_rain["No rain"],
    marker="o",
    label="No rain"
)

plt.plot(
    plot_rain.index,
    plot_rain["Rain"],
    marker="o",
    label="Rain"
)

plt.title(
    "PM2.5 During Rainy vs Dry Conditions by Hour\n"
    "(Only hours with at least 5 observations per group)"
)

plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Average PM2.5 (µg/m³)"
)

plt.xticks(
    range(24)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    figures_dir
    / "rain_vs_no_rain_by_hour.png",
    dpi=300
)

plt.close()


# =========================================================
# STEP 4:
# Print useful analytical summaries
# =========================================================

print("\nVisualisations created successfully.")


# ---------------------------------------------------------
# Highest average pollution hour
# ---------------------------------------------------------

highest_hour = hourly_pm25.idxmax()
highest_hour_value = hourly_pm25.max()

print("\nHighest average PM2.5 hour:")

print(
    f"{highest_hour}:00 "
    f"with average PM2.5 of "
    f"{highest_hour_value:.2f} µg/m³"
)


# ---------------------------------------------------------
# Lowest average pollution hour
# ---------------------------------------------------------

lowest_hour = hourly_pm25.idxmin()
lowest_hour_value = hourly_pm25.min()

print("\nLowest average PM2.5 hour:")

print(
    f"{lowest_hour}:00 "
    f"with average PM2.5 of "
    f"{lowest_hour_value:.2f} µg/m³"
)


# ---------------------------------------------------------
# Highest pollution day
# ---------------------------------------------------------

highest_day = daily_pm25.idxmax()
highest_day_value = daily_pm25.max()

print("\nHighest daily average PM2.5:")

print(
    f"{highest_day}: "
    f"{highest_day_value:.2f} µg/m³"
)


# ---------------------------------------------------------
# Rain comparison summary
# ---------------------------------------------------------

rain_difference = (
    reliable_rain["No rain"]
    -
    reliable_rain["Rain"]
)

dry_higher_count = (
    rain_difference > 0
).sum()

reliable_hour_count = len(
    reliable_rain
)

median_rain_difference = (
    rain_difference
    .median()
)

print("\nReliable rain comparison summary:")

print(
    "Hours where dry conditions had higher PM2.5:",
    dry_higher_count,
    "out of",
    reliable_hour_count
)

print(
    "Median dry-minus-rain PM2.5 difference:",
    round(
        median_rain_difference,
        2
    ),
    "µg/m³"
)


# =========================================================
# STEP 5:
# Print saved file locations
# =========================================================

print("\nSaved figures:")

print(
    figures_dir
    / "hourly_pm25_pattern.png"
)

print(
    figures_dir
    / "daily_pm25_trend.png"
)

print(
    figures_dir
    / "rain_vs_no_rain_by_hour.png"
)