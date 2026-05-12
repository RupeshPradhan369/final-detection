import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading

HF_MODEL_REPO = "dante369killer/fake-news-detector-xml-roberta"

class ModelSingleton:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        print(f"Loading model from HuggingFace: {HF_MODEL_REPO}")
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_REPO)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        return {
            'label': 'FAKE' if predicted_class == 1 else 'REAL',
            'confidence': round(confidence * 100, 2),
            'fake_probability': round(probs[0][1].item() * 100, 2),
            'real_probability': round(probs[0][0].item() * 100, 2),
        }