# api/utils/schemas.py

from pydantic import BaseModel
from typing import List, Dict


class TrainRequest(BaseModel):
    """
    Cuando envías los datos en JSON (lista de diccionarios)
    """
    data: List[Dict]


class TrainResponse(BaseModel):
    modelo: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: Dict
