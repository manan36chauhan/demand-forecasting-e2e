from pathlib import Path
import pandas as pd

import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/bike_day_features.parquet")

def main():
    # Force tracking to local folder (most predictable)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("bike_demand_daily")

    df = pd.read_parquet(DATA_PATH).sort_values("dteday")
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    features = [c for c in df.columns if c not in ["cnt", "target", "dteday"]]

    X_train, y_train = train_df[features], train_df["target"]
    X_val, y_val = val_df[features], val_df["target"]

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="xgboost_daily") as run:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)

        mlflow.log_params(params)
        mlflow.log_metric("MAE", float(mae))

        # GUARANTEE model artifact exists at artifacts/model
        mlflow.xgboost.log_model(model, name="model")

        print("Validation MAE:", mae)
        print("Run ID:", run.info.run_id)

if __name__ == "__main__":
    main()
