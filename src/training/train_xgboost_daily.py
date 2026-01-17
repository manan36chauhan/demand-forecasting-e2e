from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/bike_day_features.parquet")

def main():
    df = pd.read_parquet(DATA_PATH)

    df = df.sort_values("dteday")
    train_size = int(len(df) * 0.8)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    features = [
        c for c in df.columns
        if c not in ["cnt", "target", "dteday"]
    ]

    X_train = train_df[features]
    y_train = train_df["target"]

    X_val = val_df[features]
    y_val = val_df["target"]

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)

    print("Validation MAE:", mae)

if __name__ == "__main__":
    main()
