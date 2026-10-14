
import requests
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Thoothukudi, Tamil Nadu coordinates
LATITUDE = 8.7642
LONGITUDE = 78.1348

# NASA POWER daily weather API
url = "https://power.larc.nasa.gov/api/temporal/daily/point"

params = {
    "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN",
    "community": "AG",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": "20240101",
    "end": "20251231",
    "format": "JSON"
}

print("Downloading historical weather data...")

try:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    result = response.json()

    weather = result["properties"]["parameter"]

    df = pd.DataFrame({
       "date": pd.to_datetime(
    list(weather["PRECTOTCORR"].keys()),
    format="%Y%m%d"
),
        "rain_mm": list(weather["PRECTOTCORR"].values()),
        "avg_temp_c": list(weather["T2M"].values()),
        "tmax_c": list(weather["T2M_MAX"].values()),
        "tmin_c": list(weather["T2M_MIN"].values()),
    })

    output_file = DATA_DIR / "thoothukudi_historical_weather.csv"
    df.to_csv(output_file, index=False)

    print(f"Success! Saved {len(df)} daily records.")
    print(f"File: {output_file}")

except Exception as error:
    print("Weather download failed.")
    print("Details:", error)
