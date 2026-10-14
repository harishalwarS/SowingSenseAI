
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "thoothukudi_historical_weather.csv"

print("Loading real historical weather data...")

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required = [
    "date", "rain_mm", "avg_temp_c", "tmax_c", "tmin_c"
]

missing_columns = [col for col in required if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("\n--- Dataset validation ---")
print("Records:", len(df))
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())
print("Missing values:\n", df[required].isna().sum())

print("\n--- Historical climate summary ---")
print(df[["rain_mm", "avg_temp_c", "tmax_c", "tmin_c"]].describe().round(2))

df["month"] = df["date"].dt.month

monthly = df.groupby("month").agg(
    total_rainfall_mm=("rain_mm", "sum"),
    average_temperature_c=("avg_temp_c", "mean"),
    max_temperature_c=("tmax_c", "mean"),
    min_temperature_c=("tmin_c", "mean"),
).round(2)

output_file = BASE_DIR / "data" / "monthly_climate_summary.csv"
monthly.to_csv(output_file)

print("\n--- Monthly climate summary ---")
print(monthly)
print(f"\nSaved summary: {output_file}")
print(pd.read_csv("data/monthly_climate_summary.csv").columns.tolist())