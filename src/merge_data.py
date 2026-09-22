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

weather_path = (
    BASE_DIR
    / "data"
    / "raw"
    / "weather_raw.csv"
)
# ---------------------------------------------------------
# STEP 2: Load datasets
# ---------------------------------------------------------

air_df = pd.read_csv(
    air_quality_path,
    parse_dates=["time"]
)

weather_df = pd.read_csv(
    weather_path,
    parse_dates=["time"]
)

# ---------------------------------------------------------
# STEP 3: Validate join keys
# ---------------------------------------------------------

print("Air-quality rows:")
print(len(air_df))

print("\nWeather rows:")
print(len(weather_df))

print("\nDuplicate air-quality timestamps:")
print(air_df["time"].duplicated().sum())

print("\nDuplicate weather timestamps:")
print(weather_df["time"].duplicated().sum())

print("\nUnique air-quality timestamps:")
print(air_df["time"].nunique())

print("\nUnique weather timestamps:")
print(weather_df["time"].nunique())

# ---------------------------------------------------------
# STEP 4: Merge datasets
# ---------------------------------------------------------

merged_df = pd.merge(
    air_df,
    weather_df,
    on="time",
    how="inner",
    validate="one_to_one"
)

print("\nMerged dataset shape:")
print(merged_df.shape)

print("\nFirst five merged rows:")
print(merged_df.head())

print("\nMerged dataset information:")
merged_df.info()

print("\nMissing values:")
print(merged_df.isnull().sum())

# ---------------------------------------------------------
# STEP 5: Save analytical dataset
# ---------------------------------------------------------

output_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "kathmandu_air_weather.csv"
)

merged_df.to_csv(
    output_path,
    index=False
)

print("\nMerged dataset saved to:")
print(output_path)