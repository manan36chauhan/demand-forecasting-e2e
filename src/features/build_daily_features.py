from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/bike_day.parquet")
OUT_PATH = Path("data/processed/bike_day_features.parquet")

def main():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("dteday")

    # Target
    df["target"] = df["cnt"]

    # Lag features
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = df["cnt"].shift(lag)

    # Rolling features
    for window in [7, 14]:
        df[f"roll_mean_{window}"] = df["cnt"].shift(1).rolling(window).mean()
        df[f"roll_std_{window}"] = df["cnt"].shift(1).rolling(window).std()

    # Calendar features
    df["month"] = df["dteday"].dt.month
    df["week"] = df["dteday"].dt.isocalendar().week.astype(int)

    # Drop rows with NA (created by lags)
    df = df.dropna().reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved features to {OUT_PATH}")
    print("Shape:", df.shape)

if __name__ == "__main__":
    main()
