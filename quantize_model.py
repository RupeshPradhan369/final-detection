import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_PATH = "models/model_checkpoint"
SAVE_PATH = "models/model_quantized"

print("Loading original model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

print("Quantizing model to INT8...")
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Saving quantized model...")
os.makedirs(SAVE_PATH, exist_ok=True)

# Save quantized weights using torch.save
torch.save(quantized_model.state_dict(), f"{SAVE_PATH}/quantized_weights.pt")

# Save config and tokenizer normally
model.config.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("Done! Checking sizes...")
original_size = os.path.getsize("models/model_checkpoint/model.safetensors") / (1024*1024)
print(f"Original model size: {original_size:.1f} MB")

for f in os.listdir(SAVE_PATH):
    size = os.path.getsize(f"{SAVE_PATH}/{f}") / (1024*1024)
    print(f"{f}: {size:.1f} MB")