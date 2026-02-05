from pydantic import BaseModel, Field
from typing import Dict

FEATURE_COLUMNS = [
    'SWEAT index',
    'K index',
    'Totals totals index',
    'TLCL',
    'env_stability',
    'moisture_indices',
    'convictive_potential',
    'temp_pressure',
    'moisture_temp_profile'
]


class PredictionRequest(BaseModel):
    SWEAT_index: float = Field(..., alias="SWEAT index")
    K_index: float = Field(..., alias="K index")
    Totals_totals_index: float = Field(..., alias="Totals totals index")
    TLCL: float
    env_stability: float
    moisture_indices: float
    convictive_potential: float
    temp_pressure: float
    moisture_temp_profile: float


class PredictionResponse(BaseModel):
    probability: float
    prediction: int
