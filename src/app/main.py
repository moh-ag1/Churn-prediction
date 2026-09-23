from __future__ import annotations

from typing import Optional
import pandas as pd
import gradio as gr

from fastapi import (
    FastAPI,
    HTTPException,
)

from pydantic import BaseModel, Field

from src.serving.inference import predict


app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "Machine learning API for "
        "Telco customer churn prediction."
    ),
    version="1.0.0",
)


class CustomerData(BaseModel):

    gender: str

    SeniorCitizen: int = Field(
        ge=0,
        le=1,
    )

    Partner: str
    Dependents: str

    tenure: float = Field(
        ge=0,
    )

    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float = Field(
        ge=0,
    )

    TotalCharges: float = Field(
        ge=0,
    )


@app.get("/")
def root():

    return {
        "message": (
            "Telco Customer Churn API is running."
        ),
        "docs": "/docs",
        "ui": "/ui",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        input_data = data.model_dump()

        # Match the data types expected by the trained MLflow model
        input_data["SeniorCitizen"] = int(input_data["SeniorCitizen"])
        input_data["tenure"] = int(input_data["tenure"])
        input_data["MonthlyCharges"] = float(input_data["MonthlyCharges"])
        input_data["TotalCharges"] = float(input_data["TotalCharges"])

        result = predict(input_data)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


def gradio_predict(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges,
):
    data = {
        "gender": gender,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": int(tenure),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }

    return predict(data)

demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(
            ["Female", "Male"],
            label="Gender",
        ),

        gr.Dropdown(
            [0, 1],
            label="Senior Citizen",
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Partner",
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Dependents",
        ),

        gr.Number(
            label="Tenure",
            value=12,
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Phone Service",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No phone service",
            ],
            label="Multiple Lines",
        ),

        gr.Dropdown(
            [
                "DSL",
                "Fiber optic",
                "No",
            ],
            label="Internet Service",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Online Security",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Online Backup",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Device Protection",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Tech Support",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Streaming TV",
        ),

        gr.Dropdown(
            [
                "Yes",
                "No",
                "No internet service",
            ],
            label="Streaming Movies",
        ),

        gr.Dropdown(
            [
                "Month-to-month",
                "One year",
                "Two year",
            ],
            label="Contract",
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing",
        ),

        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
        ),

        gr.Number(
            label="Monthly Charges",
            value=70,
        ),

        gr.Number(
            label="Total Charges",
            value=840,
        ),
    ],
    outputs=gr.JSON(
        label="Prediction"
    ),
    title="Telco Customer Churn Predictor",
    description=(
        "Enter customer information to "
        "predict churn risk."
    ),
)


app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui",
)