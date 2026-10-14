
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

input_file = DATA_DIR / "thoothukudi_historical_weather.csv"
output_file = DATA_DIR / "sowing_window_features.csv"

df = pd.read_csv(input_file, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Create rolling 7-day climate features
df["rain_7day_mm"] = df["rain_mm"].rolling(7).sum()
df["avg_temp_7day_c"] = df["avg_temp_c"].rolling(7).mean()
df["max_temp_7day_c"] = df["tmax_c"].rolling(7).mean()
df["min_temp_7day_c"] = df["tmin_c"].rolling(7).mean()

# Count consecutive dry days (< 1 mm rainfall)
dry_spell = []
current_dry_days = 0

for rainfall in df["rain_mm"]:
    if rainfall < 1.0:
        current_dry_days += 1
    else:
        current_dry_days = 0
    dry_spell.append(current_dry_days)

df["dry_spell_days"] = dry_spell

features = df.dropna().copy()
features.to_csv(output_file, index=False)

print("7-day sowing-window features created!")
print("Rows:", len(features))
print("Saved:", output_file)
print(features.tail(10).round(2).to_string(index=False))
