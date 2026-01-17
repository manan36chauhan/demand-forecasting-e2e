from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/bike_sharing")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    day_path = RAW_DIR / "day.csv"
    hour_path = RAW_DIR / "hour.csv"

    if not day_path.exists() or not hour_path.exists():
        raise FileNotFoundError(
            "Dataset files not found. Expected day.csv and hour.csv in data/raw/bike_sharing/"
        )

    df_day = pd.read_csv(day_path)
    df_hour = pd.read_csv(hour_path)

    # Basic sanity checks
    print("DAY SHAPE:", df_day.shape)
    print("HOUR SHAPE:", df_hour.shape)
    print("DAY COLUMNS:", list(df_day.columns))
    print("HOUR COLUMNS:", list(df_hour.columns))

    # Parse dates
    if "dteday" in df_day.columns:
        df_day["dteday"] = pd.to_datetime(df_day["dteday"])
    if "dteday" in df_hour.columns:
        df_hour["dteday"] = pd.to_datetime(df_hour["dteday"])

    # Save as parquet (faster + standard for analytics)
    df_day.to_parquet(OUT_DIR / "bike_day.parquet", index=False)
    df_hour.to_parquet(OUT_DIR / "bike_hour.parquet", index=False)

    print("Saved:")
    print(" -", OUT_DIR / "bike_day.parquet")
    print(" -", OUT_DIR / "bike_hour.parquet")

if __name__ == "__main__":
    main()
