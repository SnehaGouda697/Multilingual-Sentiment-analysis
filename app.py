from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from pydantic import BaseModel
from model.hybrid_model import HybridSentimentModel
from utils.preprocess import clean_text

app = FastAPI()

model = HybridSentimentModel()

class InputText(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontened/index.html", "r",encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(data: InputText):
    cleaned = clean_text(data.text)
    result = model.predict(cleaned)
    return result