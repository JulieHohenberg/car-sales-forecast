
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from xgboost import XGBRegressor

# load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.json")
model = XGBRegressor()
model.load_model(model_path)

# initialize FastAPI app
app = FastAPI()

# allow frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# define request body schema
class SalesInput(BaseModel):
    sales: list[float]  # Expect exactly 6 values

# mount the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# serve index.html at root
@app.get("/")
def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# define the prediction endpoint
@app.post("/predict")
async def predict(input: SalesInput):

    # check that exactly 6 months were provided
    if len(input.sales) != 6:
        return {"error": "Exactly 6 months of sales data required."}

    # convert sales to DataFrame
    data = pd.DataFrame([input.sales], columns=[
        'lag_1', 'lag_2', 'lag_3',
        'lag_4', 'lag_5', 'lag_6'
    ])

    # predict next month's sales
    prediction = model.predict(data)[0]

    return {"prediction": float(prediction)}
