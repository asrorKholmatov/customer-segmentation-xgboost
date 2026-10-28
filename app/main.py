"""REST API for integrating the segment classifier into other systems (CRM, ETL jobs).

Run:  uvicorn app.main:app --reload
"""
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.model import predict

app = FastAPI(title="SegmentIQ API", version="1.0.0",
              description="XGBoost customer segmentation (segments A-D).")

YesNo = Optional[Literal["Yes", "No"]]


class Customer(BaseModel):
    Gender: Literal["Male", "Female"]
    Ever_Married: YesNo = None
    Age: int = Field(ge=18, le=100)
    Graduated: YesNo = None
    Profession: Optional[str] = None
    Work_Experience: Optional[float] = Field(None, ge=0)
    Spending_Score: Literal["Low", "Average", "High"]
    Family_Size: Optional[float] = Field(None, ge=1)
    Var_1: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_one(customer: Customer):
    return predict([customer.model_dump()])[0]


@app.post("/predict/batch")
def predict_batch(customers: list[Customer]):
    return predict([c.model_dump() for c in customers])
