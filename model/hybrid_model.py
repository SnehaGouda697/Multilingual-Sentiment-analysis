from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class HybridSentimentModel:
    def __init__(self):
        # Models
        self.mt5_model_name = "google/mt5-small"
        self.xlmr_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.indicbert_model_name = "ai4bharat/indic-bert"

        # Load XLM-R (main working model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.xlmr_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.xlmr_model_name)

        self.labels = ["negative", "neutral", "positive"]

    def predict_xlmr(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        return self.labels[pred.item()], float(confidence.item())

    def predict(self, text):
        label, confidence = self.predict_xlmr(text)

        # Hybrid logic (basic)
        if confidence < 0.6:
            # fallback logic (can extend later)
            label = label  # placeholder

        return {
            "label": label,
            "confidence": round(confidence, 4)
        }