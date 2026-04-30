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

        text_lower = text.lower()

        # 🔹 Rule-based domain logic (EV specific)
        if "battery" in text_lower:
            if "bad" in text_lower or "poor" in text_lower:
                return {"label": "negative", "confidence": confidence}
            elif "good" in text_lower or "excellent" in text_lower:
                return {"label": "positive", "confidence": confidence}

        if "expensive" in text_lower or "costly" in text_lower:
            return {"label": "negative", "confidence": confidence}

        if "cheap" in text_lower or "affordable" in text_lower:
            return {"label": "positive", "confidence": confidence}

        # 🔹 Confidence fallback
        if confidence < 0.6:
            if "not" in text_lower:
                return {"label": "negative", "confidence": confidence}

        return {
            "label": label,
            "confidence": round(confidence, 4)
        }