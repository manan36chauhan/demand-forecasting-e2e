from typing import Any, Dict
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import mlflow
from mlflow.tracking import MlflowClient

app = FastAPI(title="Bike Demand Forecast API", version="1.0")

# Use your SQLite tracking DB (since you have mlflow.db)
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "bike_demand_daily"

MODEL = None

class PredictRequest(BaseModel):
    season: float
    yr: float
    mnth: float
    holiday: float
    weekday: float
    workingday: float
    weathersit: float
    temp: float
    atemp: float
    hum: float
    windspeed: float
    lag_1: float
    lag_7: float
    lag_14: float
    roll_mean_7: float
    roll_std_7: float
    roll_mean_14: float
    roll_std_14: float
    month: float
    week: float

@app.on_event("startup")
def load_model() -> None:
    global MODEL
    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise FileNotFoundError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found. Run training first."
        )

    # Get latest run by start_time
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError("No MLflow runs found. Run training first.")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    MODEL = mlflow.pyfunc.load_model(model_uri)
    print("Loaded model from:", model_uri)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    if MODEL is None:
        return {"error": "Model not loaded"}

    X = pd.DataFrame([req.model_dump()])
    yhat = float(MODEL.predict(X)[0])
    return {"prediction_cnt": yhat}
