from fastapi import FastAPI
from pydantic import BaseModel
from model.hybrid_model import HybridSentimentModel
from utils.preprocess import clean_text

app = FastAPI()

model = HybridSentimentModel()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "EV Sentiment API Running"}

@app.post("/predict")
def predict(data: InputText):
    cleaned = clean_text(data.text)
    result = model.predict(cleaned)
    return result