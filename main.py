from fastapi import FastAPI
from fastapi.responses import FileResponse
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Real-World Sales Prediction API")
model = joblib.load("sales_model.pkl")


class AdBudget(BaseModel):
    TV: float
    Radio: float
    Newspaper: float


# હોમ પેજ પર સીધું HTML UI બતાવવા માટે
@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict_sales(budget: AdBudget):
    input_data = pd.DataFrame(
        [[budget.TV, budget.Radio, budget.Newspaper]],
        columns=["TV", "Radio", "Newspaper"],
    )
    prediction = model.predict(input_data)
    return {
        "input_budget": budget,
        "predicted_sales": round(float(prediction[0]), 2),
    }