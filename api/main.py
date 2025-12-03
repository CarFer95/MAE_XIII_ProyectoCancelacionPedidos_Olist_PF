# api/main.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd

from api.utils.schemas import TrainRequest, TrainResponse
from api.utils.training_api import train_model_from_df


app = FastAPI(
    title="API Cancelaciones Olist",
    description="Entrena el modelo XGB Tuned con datos enviados vía API",
    version="1.0"
)


# =========================
# 1) Entrenar desde un CSV
# =========================
@app.post("/train/csv", response_model=TrainResponse)
async def train_from_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    result = train_model_from_df(df)
    return result


# =========================
# 2) Entrenar desde JSON
# =========================
@app.post("/train/json", response_model=TrainResponse)
async def train_from_json(payload: TrainRequest):
    df = pd.DataFrame(payload.data)
    result = train_model_from_df(df)
    return result


# =========================
# 3) Endpoint básico
# =========================
@app.get("/")
def root():
    return {"status": "API OK", "version": "1.0"}
